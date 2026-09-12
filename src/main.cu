#include "kernels.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) \
  throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e)); } while (0)

struct Options { std::string op = "all"; int m = 512, n = 512, k = 512, rows = 1024,
  cols = 768, warmup = 10, iterations = 100; bool check = false; };

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto value = [&]() -> std::string { if (++i >= argc) throw std::runtime_error("missing value for " + a); return argv[i]; };
    if (a == "--op") o.op = value(); else if (a == "--m") o.m = std::stoi(value());
    else if (a == "--n") o.n = std::stoi(value()); else if (a == "--k") o.k = std::stoi(value());
    else if (a == "--rows") o.rows = std::stoi(value()); else if (a == "--cols") o.cols = std::stoi(value());
    else if (a == "--warmup") o.warmup = std::stoi(value()); else if (a == "--iterations") o.iterations = std::stoi(value());
    else if (a == "--check") o.check = true; else if (a == "--help") {
      std::cout << "Usage: transformer_ops [--op all|matmul|softmax|layernorm] [--m N --n N --k N]\n"
                   "  [--rows N --cols N] [--warmup N --iterations N] [--check]\n"; std::exit(0);
    } else throw std::runtime_error("unknown argument: " + a);
  }
  if (o.m <= 0 || o.n <= 0 || o.k <= 0 || o.rows <= 0 || o.cols <= 0 || o.iterations <= 0 || o.warmup < 0)
    throw std::runtime_error("dimensions/iterations must be positive and warmup nonnegative");
  if (o.op != "all" && o.op != "matmul" && o.op != "softmax" && o.op != "layernorm")
    throw std::runtime_error("--op must be all, matmul, softmax, or layernorm");
  return o;
}

std::vector<float> random_vector(size_t size, int seed) {
  std::mt19937 gen(seed); std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(size); for (float& x : v) x = dist(gen); return v;
}

template <typename Launch> float gpu_ms(Launch launch, int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start)); for (int i = 0; i < iterations; ++i) launch(); CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop)); float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop)); return ms / iterations;
}

float max_error(const std::vector<float>& got, const std::vector<float>& ref) {
  float e = 0; for (size_t i = 0; i < got.size(); ++i) e = std::max(e, std::abs(got[i] - ref[i])); return e;
}

void report(const std::string& op, float gpu, double cpu, float error, bool checked) {
  std::cout << std::left << std::setw(12) << op << " GPU " << std::fixed << std::setprecision(4) << gpu
            << " ms";
  if (checked) std::cout << " | CPU " << cpu << " ms | speedup " << cpu / gpu
                         << "x | max error " << std::scientific << error;
  std::cout << '\n';
}

void run_matmul(const Options& o) {
  auto a = random_vector(size_t(o.m) * o.k, 1), b = random_vector(size_t(o.k) * o.n, 2);
  std::vector<float> out(size_t(o.m) * o.n), ref(out.size()); float *da, *db, *dc;
  CUDA_CHECK(cudaMalloc(&da, a.size()*sizeof(float))); CUDA_CHECK(cudaMalloc(&db, b.size()*sizeof(float))); CUDA_CHECK(cudaMalloc(&dc, out.size()*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(da, a.data(), a.size()*sizeof(float), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(db, b.data(), b.size()*sizeof(float), cudaMemcpyHostToDevice));
  auto launch = [&]{ transformer_ops::launch_matmul(da, db, dc, o.m, o.n, o.k); };
  float ms = gpu_ms(launch, o.warmup, o.iterations); double cpu = 0; float err = 0;
  if (o.check) { auto t = std::chrono::steady_clock::now(); for (int i=0;i<o.m;++i) for(int j=0;j<o.n;++j) { float s=0; for(int p=0;p<o.k;++p) s += a[i*o.k+p]*b[p*o.n+j]; ref[i*o.n+j]=s; } cpu=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count(); CUDA_CHECK(cudaMemcpy(out.data(),dc,out.size()*sizeof(float),cudaMemcpyDeviceToHost)); err=max_error(out,ref); if(err > 2e-3f*o.k) throw std::runtime_error("matmul correctness check failed"); }
  report("matmul", ms, cpu, err, o.check); CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
}

void run_rowwise(const Options& o, bool layernorm) {
  auto x=random_vector(size_t(o.rows)*o.cols,3), gamma=random_vector(o.cols,4), beta=random_vector(o.cols,5);
  std::vector<float> out(x.size()), ref(x.size()); float *dx,*dy,*dg=nullptr,*db=nullptr;
  CUDA_CHECK(cudaMalloc(&dx,x.size()*sizeof(float))); CUDA_CHECK(cudaMalloc(&dy,x.size()*sizeof(float))); CUDA_CHECK(cudaMemcpy(dx,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice));
  if(layernorm){ CUDA_CHECK(cudaMalloc(&dg,gamma.size()*sizeof(float))); CUDA_CHECK(cudaMalloc(&db,beta.size()*sizeof(float))); CUDA_CHECK(cudaMemcpy(dg,gamma.data(),gamma.size()*sizeof(float),cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(db,beta.data(),beta.size()*sizeof(float),cudaMemcpyHostToDevice)); }
  auto launch=[&]{ if(layernorm) transformer_ops::launch_layernorm(dx,dg,db,dy,o.rows,o.cols); else transformer_ops::launch_softmax(dx,dy,o.rows,o.cols); };
  float ms=gpu_ms(launch,o.warmup,o.iterations); double cpu=0; float err=0;
  if(o.check){ auto t=std::chrono::steady_clock::now(); for(int r=0;r<o.rows;++r){ if(layernorm){ double mean=0,sq=0; for(int c=0;c<o.cols;++c) mean+=x[r*o.cols+c]; mean/=o.cols; for(int c=0;c<o.cols;++c){double d=x[r*o.cols+c]-mean;sq+=d*d;} double inv=1/std::sqrt(sq/o.cols+1e-5); for(int c=0;c<o.cols;++c) ref[r*o.cols+c]=(x[r*o.cols+c]-mean)*inv*gamma[c]+beta[c]; } else { float mx=-INFINITY; for(int c=0;c<o.cols;++c) mx=std::max(mx,x[r*o.cols+c]); double sum=0; for(int c=0;c<o.cols;++c) sum+=std::exp(x[r*o.cols+c]-mx); for(int c=0;c<o.cols;++c) ref[r*o.cols+c]=std::exp(x[r*o.cols+c]-mx)/sum; }} cpu=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count(); CUDA_CHECK(cudaMemcpy(out.data(),dy,out.size()*sizeof(float),cudaMemcpyDeviceToHost)); err=max_error(out,ref); if(err>2e-4f) throw std::runtime_error(std::string(layernorm?"layernorm":"softmax")+" correctness check failed"); }
  report(layernorm?"layernorm":"softmax",ms,cpu,err,o.check); CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy)); if(layernorm){CUDA_CHECK(cudaFree(dg));CUDA_CHECK(cudaFree(db));}
}

int main(int argc,char** argv){ try { auto o=parse(argc,argv); int devices=0; CUDA_CHECK(cudaGetDeviceCount(&devices)); if(!devices) throw std::runtime_error("no CUDA device found"); cudaDeviceProp p{}; CUDA_CHECK(cudaGetDeviceProperties(&p,0)); std::cout<<"Device: "<<p.name<<"\n"; if(o.op=="all"||o.op=="matmul") run_matmul(o); if(o.op=="all"||o.op=="softmax") run_rowwise(o,false); if(o.op=="all"||o.op=="layernorm") run_rowwise(o,true); return 0; } catch(const std::exception& e){ std::cerr<<"error: "<<e.what()<<'\n'; return 1; } }
