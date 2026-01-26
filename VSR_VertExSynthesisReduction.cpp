/*
 * Tipo: code/cpp (único archivo con implementación y ejemplo de uso)
 * Propósito: Proveer una implementación de referencia, documentada y lista
 *            para integrar en un motor C++ como módulo de reducción adaptativa
 *            de vértices.
 *
 * NOTAS:
 * - Este archivo es una referencia conceptual y contiene implementaciones
 *   prácticas y portables que funcionan en CPU. Se incluyen hooks y puntos
 *   donde se puede desplazar trabajo a SPU/VU/compute shaders para plataformas
 *   específicas (PS3/PS2/GPU).
 * - Está pensado para ser seguro, modular y con puntos de extensión.
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace tx::vsr {

// ----------------------------- Basic types ---------------------------------

struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}
    Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
};

inline float dot(const Vec3& a, const Vec3& b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b){ return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x }; }
inline float len(const Vec3& v){ return std::sqrt(dot(v,v)); }
inline Vec3 normalize(const Vec3& v){ float L=len(v); return L>0 ? v*(1.0f/L) : v; }

struct Vertex {
    Vec3 p;
    Vec3 n;
    float u, v;
    // extra attributes can be added (tangents, colors...)
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices; // triangles (tri0, tri1, tri2 ...)

    size_t triangle_count() const { return indices.size() / 3; }
};

// -------------------------- Configuration / Policies -----------------------

struct VSRConfig {
    // Target reduction ratio (0.0 = no reduction, 1.0 = aggressive)
    float reduction_ratio = 0.75f;

    // Minimal number of vertices to keep (safety)
    size_t min_vertices = 64;

    // Importance falloff radius for locality heuristics
    float locality_radius = 2.0f;

    // Perceptual budget threshold (0..1) - controls allowed error
    float perceptual_budget = 0.05f;

    // Tile size for tiled processing (screen-space / cluster-space)
    int tile_size = 64;

    // Maximum cpu threads used for parallel passes
    int max_threads = std::thread::hardware_concurrency() > 0 ? (int)std::thread::hardware_concurrency() : 4;

    // Random seed for stochastic synthesis
    uint32_t rng_seed = 0xC0FFEE;
};

// ------------------------------- Utilities ---------------------------------

static inline void clampf(float& v, float a=0.0f, float b=1.0f){ if(v<a) v=a; if(v>b) v=b; }

// Simple triangle area
inline float triangle_area(const Vec3& a, const Vec3& b, const Vec3& c) {
    return 0.5f * len(cross(b-a, c-a));
}

// -------------------------- Importance Estimator ---------------------------

// Per-vertex importance score (0..1). High = keep high fidelity.
// The estimator combines geometric saliency (curvature), screen-space influence
// (approx via normal facing or custom weight), motion/velocity hints, and
// material/semantic weights provided externally.

struct ImportanceField {
    std::vector<float> scores; // per-vertex

    ImportanceField(size_t n = 0) : scores(n, 0.0f) {}

    void resize(size_t n){ scores.assign(n, 0.0f); }

    float& operator[](size_t i){ return scores[i]; }
    const float& operator[](size_t i) const{ return scores[i]; }
};

class ImportanceEstimator {
public:
    ImportanceEstimator(const VSRConfig& cfg) : cfg(cfg) {}

    // Compute base importance per-vertex using simple curvature proxy & area-weighted normal
    ImportanceField estimate(const Mesh& mesh) const {
        size_t nv = mesh.vertices.size();
        ImportanceField f(nv);
        // accumulate adjacent triangle normals and angles
        std::vector<float> areaAccum(nv, 0.0f);
        std::vector<Vec3> normalAccum(nv, {0,0,0});

        for(size_t t=0;t<mesh.triangle_count();++t){
            uint32_t i0 = mesh.indices[3*t+0];
            uint32_t i1 = mesh.indices[3*t+1];
            uint32_t i2 = mesh.indices[3*t+2];
            Vec3 p0 = mesh.vertices[i0].p;
            Vec3 p1 = mesh.vertices[i1].p;
            Vec3 p2 = mesh.vertices[i2].p;
            float A = triangle_area(p0,p1,p2);
            Vec3 triN = normalize(cross(p1-p0, p2-p0));
            normalAccum[i0] = normalAccum[i0] + triN * A;
            normalAccum[i1] = normalAccum[i1] + triN * A;
            normalAccum[i2] = normalAccum[i2] + triN * A;
            areaAccum[i0] += A; areaAccum[i1]+=A; areaAccum[i2]+=A;
        }
        // curvature proxy: difference between vertex normal and averaged tri normals
        for(size_t i=0;i<nv;++i){
            Vec3 avgN = areaAccum[i] > 0 ? normalAccum[i] * (1.0f/areaAccum[i]) : mesh.vertices[i].n;
            float curv = 1.0f - dot(normalize(avgN), normalize(mesh.vertices[i].n));
            // area-based weight to prefer larger triangles preserved
            float areaW = std::min(1.0f, areaAccum[i] * 10.0f);
            float base = clamp01(curv * 0.7f + areaW * 0.3f);
            f[i] = base;
        }
        // normalize to 0..1
        normalize_field(f);
        return f;
    }

private:
    const VSRConfig& cfg;

    static float clamp01(float x){ if(x<0.0f) return 0.0f; if(x>1.0f) return 1.0f; return x; }

    static void normalize_field(ImportanceField& f){
        float mx = 0.0f; for(float s : f.scores) if(s>mx) mx=s;
        if(mx<=0.0f) return;
        for(float &s : f.scores) s/=mx;
    }
};

// -------------------------- Synthetic Vertex Generator ---------------------

// The heart of VSR: given a mesh and importance field, synthesize a reduced mesh.
// Approach (composite):
// 1. Compute target vertex budget.
// 2. Sample / keep high-importance vertices.
// 3. For low-importance regions, generate a "synthesized" representation using
//    - stochastic collapsing (vertex clusters -> representative vertex)
//    - procedural surface patch reconstruction (local barycentric resampling)
// 4. Rebuild indices and optionally produce simplified triangles or patch proxies.

class VertexSynthesizer {
public:
    VertexSynthesizer(const VSRConfig& cfg) : cfg(cfg), rng(cfg.rng_seed) {}

    // Primary API: produce reduced mesh with mapping data
    Mesh synthesize(const Mesh& src, const ImportanceField& imp) {
        assert(src.vertices.size() == imp.scores.size());
        Mesh out;
        size_t srcV = src.vertices.size();

        // determine target vertex count
        size_t targetV = std::max(cfg.min_vertices, (size_t)(srcV * (1.0f - cfg.reduction_ratio)));
        targetV = std::min(targetV, srcV);

        // 1. select keepers by thresholding importance
        std::vector<uint8_t> keeperFlags(srcV, 0);
        float threshold = choose_threshold(imp, targetV);
        for(size_t i=0;i<srcV;++i) if(imp.scores[i] >= threshold) keeperFlags[i]=1;

        size_t kept = std::accumulate(keeperFlags.begin(), keeperFlags.end(), (size_t)0);

        // If too many kept (rare), raise threshold
        if(kept > targetV){
            threshold = raise_threshold_to_count(imp, targetV);
            std::fill(keeperFlags.begin(), keeperFlags.end(), 0);
            for(size_t i=0;i<srcV;++i) if(imp.scores[i] >= threshold) keeperFlags[i]=1;
            kept = std::accumulate(keeperFlags.begin(), keeperFlags.end(), (size_t)0);
        }

        // 2. cluster remaining vertices by spatial locality to produce representatives
        std::vector<int> clusterIndex(srcV, -1);
        int nextClusterId = 0;

        // create spatial grid for clustering
        float cellSize = cfg.locality_radius; if(cellSize<=0) cellSize = 1.0f;
        std::unordered_map<int64_t, std::vector<size_t>> grid;
        grid.reserve(srcV/8 + 1);
        for(size_t i=0;i<srcV;++i){
            if(keeperFlags[i]) continue;
            auto key = hash_cell(src.vertices[i].p, cellSize);
            grid[key].push_back(i);
        }
        // generate clusters per cell
        std::vector<std::vector<size_t>> clusters;
        clusters.reserve(grid.size());
        for(auto &kv : grid){
            // optionally split large cells into subclusters by k-means or random partition
            const std::vector<size_t>& cellVerts = kv.second;
            if(cellVerts.empty()) continue;
            // create one cluster per cell (fast) — can be refined
            clusters.emplace_back(cellVerts);
        }

        // 3. build output vertices: first keepers (copy), then cluster representatives
        std::vector<int> srcToOutIndex(srcV, -1);
        out.vertices.reserve( kept + clusters.size() );
        for(size_t i=0;i<srcV;++i){
            if(keeperFlags[i]){
                srcToOutIndex[i] = (int)out.vertices.size();
                out.vertices.push_back(src.vertices[i]);
            }
        }
        // cluster reps
        for(const auto &cl : clusters){
            Vertex rep = synthesize_cluster_rep(src, cl);
            int outIdx = (int)out.vertices.size();
            out.vertices.push_back(rep);
            for(size_t sidx : cl) srcToOutIndex[sidx] = outIdx;
        }

        // 4. rebuild triangles: for each source triangle, map vertices to representatives
        out.indices.reserve(src.indices.size());
        for(size_t t=0;t<src.triangle_count();++t){
            uint32_t i0 = src.indices[3*t+0];
            uint32_t i1 = src.indices[3*t+1];
            uint32_t i2 = src.indices[3*t+2];
            int n0 = srcToOutIndex[i0];
            int n1 = srcToOutIndex[i1];
            int n2 = srcToOutIndex[i2];
            // degenerate triangles (collapsed to a line or point) are skipped
            if(n0<0 || n1<0 || n2<0) continue;
            if(n0==n1 || n1==n2 || n2==n0) continue;
            out.indices.push_back((uint32_t)n0);
            out.indices.push_back((uint32_t)n1);
            out.indices.push_back((uint32_t)n2);
        }

        // 5. optional post-process: planar patch reconstruction, remap normals, weld vertices
        weld_vertices(out, 1e-6f);
        recalc_normals_if_missing(out);

        return out;
    }

private:
    const VSRConfig& cfg;
    mutable std::mt19937 rng;

    static inline int64_t combine64(uint32_t a, uint32_t b){ return ((int64_t)a<<32) | b; }

    static int64_t hash_cell(const Vec3& p, float cellSize){
        int32_t xi = (int32_t)std::floor(p.x / cellSize);
        int32_t yi = (int32_t)std::floor(p.y / cellSize);
        int32_t zi = (int32_t)std::floor(p.z / cellSize);
        // pack into 64 bits (note: collisions possible but unlikely)
        uint32_t h1 = (uint32_t)(xi*73856093 ^ yi*19349663);
        uint32_t h2 = (uint32_t)zi*83492791;
        return combine64(h1,h2);
    }

    float choose_threshold(const ImportanceField& imp, size_t targetV) const {
        // pick percentile: we want approximately targetV vertices kept by threshold
        std::vector<float> s = imp.scores;
        std::sort(s.begin(), s.end(), std::greater<float>());
        size_t nv = s.size();
        if(targetV==0 || targetV>=nv) return 1.0f; // keep all
        size_t idx = std::min(nv-1, targetV);
        return s[idx];
    }

    float raise_threshold_to_count(const ImportanceField& imp, size_t target) const {
        std::vector<float> s = imp.scores;
        std::sort(s.begin(), s.end(), std::greater<float>());
        size_t nv = s.size();
        if(target==0) return 1.0f;
        size_t idx = std::min(nv-1, target-1);
        return s[idx];
    }

    Vertex synthesize_cluster_rep(const Mesh& src, const std::vector<size_t>& cluster) {
        // compute area-weighted centroid
        Vec3 pavg{0,0,0}; Vec3 navg{0,0,0}; float totalA=0.0f;
        // naive: average positions and normals, fallback to random member
        for(size_t vi : cluster){
            pavg = pavg + src.vertices[vi].p;
            navg = navg + src.vertices[vi].n;
            totalA += 1.0f;
        }
        float inv = 1.0f / std::max(1.0f, totalA);
        Vertex rep;
        rep.p = pavg * inv;
        rep.n = normalize(navg * inv);
        rep.u = rep.v = 0.0f; // UVs could be derived via local parametrization (omitted for brevity)

        // small stochastic perturbation to avoid T-vertex artifacts
        std::uniform_real_distribution<float> dist(-1e-4f, 1e-4f);
        rep.p.x += dist(rng); rep.p.y += dist(rng); rep.p.z += dist(rng);
        return rep;
    }

    static void weld_vertices(Mesh& m, float eps){
        // naive O(n^2) weld — replace with hash grid for production
        size_t n = m.vertices.size();
        std::vector<int> map(n); iota(map.begin(), map.end(), 0);
        for(size_t i=0;i<n;++i){
            if(map[i]!=(int)i) continue;
            for(size_t j=i+1;j<n;++j){
                if(map[j]!=(int)j) continue;
                if(distance_sq(m.vertices[i].p, m.vertices[j].p) <= eps*eps){
                    map[j] = map[i];
                }
            }
        }
        // remap indices
        for(uint32_t &idx : m.indices) idx = (uint32_t)map[idx];
        // compact vertices
        std::vector<Vertex> nv; nv.reserve(n);
        std::vector<int> newIndex(n, -1);
        for(size_t i=0;i<n;++i){ if(map[i]==(int)i){ newIndex[i]=(int)nv.size(); nv.push_back(m.vertices[i]); } }
        for(size_t i=0;i<n;++i) if(map[i]!=(int)i) newIndex[i] = newIndex[map[i]];
        for(uint32_t &idx : m.indices) idx = (uint32_t)newIndex[idx];
        m.vertices.swap(nv);
    }

    static float distance_sq(const Vec3& a, const Vec3& b){ float dx=a.x-b.x; float dy=a.y-b.y; float dz=a.z-b.z; return dx*dx+dy*dy+dz*dz; }

    static void recalc_normals_if_missing(Mesh& m){
        // compute per-vertex normals by area-weighted tri normals
        size_t nv = m.vertices.size();
        std::vector<Vec3> accum(nv,{0,0,0});
        std::vector<float> area(nv,0.0f);
        for(size_t t=0;t<m.triangle_count();++t){
            uint32_t i0 = m.indices[3*t+0];
            uint32_t i1 = m.indices[3*t+1];
            uint32_t i2 = m.indices[3*t+2];
            Vec3 p0=m.vertices[i0].p, p1=m.vertices[i1].p, p2=m.vertices[i2].p;
            float A = triangle_area(p0,p1,p2);
            Vec3 triN = normalize(cross(p1-p0,p2-p0));
            accum[i0] = accum[i0] + triN * A; area[i0]+=A;
            accum[i1] = accum[i1] + triN * A; area[i1]+=A;
            accum[i2] = accum[i2] + triN * A; area[i2]+=A;
        }
        for(size_t i=0;i<nv;++i){ m.vertices[i].n = area[i]>0 ? normalize(accum[i] * (1.0f/area[i])) : Vec3{0,1,0}; }
    }
};

// -------------------------- VSR Manager (High-level) -----------------------

class VSR {
public:
    VSR(const VSRConfig& cfg) : cfg(cfg), estimator(cfg), synth(cfg) {}

    // Run full pipeline: estimate importance, synthesize reduced mesh
    Mesh process(const Mesh& src) {
        ImportanceField imp = estimator.estimate(src);

        // optional: inject external hints (visibility / material importance)
        // ... (hook)

        Mesh reduced = synth.synthesize(src, imp);

        // optional: multi-pass refinement (morphing prototypes to match error budget)
        // ... (hook)

        return reduced;
    }

private:
    VSRConfig cfg;
    ImportanceEstimator estimator;
    VertexSynthesizer synth;
};

// ----------------------------- Example usage --------------------------------

// simple mesh generator (icosahedron-like) for demo purposes
Mesh create_test_mesh_grid(int w, int h){
    Mesh m;
    m.vertices.reserve(w*h);
    for(int y=0;y<h;++y){
        for(int x=0;x<w;++x){
            Vertex v; v.p = Vec3((float)x, (float)y, std::sin((float)x*0.1f)*0.2f);
            v.n = Vec3(0,0,1);
            v.u = x/(float)w; v.v = y/(float)h;
            m.vertices.push_back(v);
        }
    }
    for(int y=0;y<h-1;++y){
        for(int x=0;x<w-1;++x){
            uint32_t i0 = y*w + x;
            uint32_t i1 = y*w + (x+1);
            uint32_t i2 = (y+1)*w + x;
            uint32_t i3 = (y+1)*w + (x+1);
            // two tris
            m.indices.push_back(i0); m.indices.push_back(i2); m.indices.push_back(i1);
            m.indices.push_back(i1); m.indices.push_back(i2); m.indices.push_back(i3);
        }
    }
    return m;
}

int main(){
    std::cout << "VSR Demo: Vertex Synthesis Reduction (conceptual)\n";
    Mesh src = create_test_mesh_grid(120, 80); // ~9600 vertices
    std::cout << "Source vertices: " << src.vertices.size() << " tris: " << src.triangle_count() << "\n";

    VSRConfig cfg;
    cfg.reduction_ratio = 0.85f; // large reduction
    cfg.min_vertices = 256;
    cfg.locality_radius = 3.0f;
    cfg.rng_seed = 12345;

    VSR vsr(cfg);
    Mesh reduced = vsr.process(src);

    std::cout << "Reduced vertices: " << reduced.vertices.size() << " tris: " << reduced.triangle_count() << "\n";

    // Note: In a real engine you would now upload reduced mesh to GPU and integrate with
    // VUART/ATDC pipelines (e.g., generate indirect draw calls, attach LODs, etc.).

    return 0;

}
