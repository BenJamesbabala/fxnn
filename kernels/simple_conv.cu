#define O_CHANNELS 5

void __global__ sconv_packed_fwd_row_f32_ker(
        float *v_dst, float *v_src, int w, int h, int bsize) {
    /*
       ALWAYS ASSUME:
            w,h both less or equal than 512 and is power of 2
     */
    const int x = threadIdx.x;
    const int y = threadIdx.y + blockIdx.y*blockDim.y;
    const int z = threadIdx.z + blockIdx.z*blockDim.z;
    const int by = blockDim.y;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;

    if(x>=w) return;
    if(y>=h) return;
    if(z>=bsize) return;

    extern __shared__ float _inp[];

    float *src_base = v_src+z*w*h;
    float *inp = _inp + (ty+tz*by)*(w+2);
    inp[x+1] = src_base[x+y*w];
    inp[0] = 0.f;
    inp[w+1] = 0.f;

    __syncthreads();
    float *dst_base = v_dst+z*O_CHANNELS*w*h;
    dst_base[x+y*w] = inp[x+2] - inp[x];

    dst_base[x+(y+h)*w] = inp[x+2]+2*inp[x+1]+inp[x];
}

void __global__ sconv_packed_fwd_col_f32_ker(
        float *v_dst, float *v_src, int w, int h, int bsize) {
    /*
       ALWAYS ASSUME:
            w,h both less or equal than 512 and is power of 2
     */
    const int x = threadIdx.x + blockIdx.x*blockDim.x;
    const int y = threadIdx.y;
    const int z = threadIdx.z + blockIdx.z*blockDim.z;
    const int bx = blockDim.x;
    const int tx = threadIdx.x;
    const int tz = threadIdx.z;

    if(x>=w) return;
    if(y>=h) return;
    if(z>=bsize) return;

    extern __shared__ float _inp[];
    float *inp = _inp + (tx+tz*bx)*2*(h+2);
    float *inp2 = inp+h+2;

    float *dst_base = v_dst+z*O_CHANNELS*w*h;
    inp[y+1] = dst_base[x+y*h];
    inp2[y+1]= dst_base[x+(y+w)*h];
    inp[0] = 0.f;
    inp2[0] = 0.f;
    inp[h+1] = 0.f;
    inp2[h+1] = 0.f;

    __syncthreads();
    dst_base[x+y*w] = v_src[x+(y+z*h)*w];
    dst_base[x+(y+h)*w] = inp[y+2]-inp[y];
    dst_base[x+(y+h*3)*w] = inp2[y+2]-inp2[y];

    dst_base[x+(y+h*2)*w] = inp[y+2]+inp[y+1]*2+inp[y];
    dst_base[x+(y+h*4)*w] = inp2[y+2]+inp2[y+1]*2+inp2[y]; }

void __global__ sconv_packed_bwd_col_f32_ker(
        float *v_dedx, float *v_tmp, float *v_dedy, int w, int h, int bsize) {
    const int x = threadIdx.x + blockIdx.x*blockDim.x;
    const int y = threadIdx.y;
    const int z = threadIdx.z + blockIdx.z*blockDim.z;
    const int bx = blockDim.x;
    const int tx = threadIdx.x;
    const int tz = threadIdx.z;

    if(x>=w) return;
    if(y>=h) return;
    if(z>=bsize) return;

    extern __shared__ float _inp[];
    float *inp0 = _inp + (tx+tz*bx)*4*(h+2);
    float *inp1 = inp0+h+2;
    float *inp2 = inp0+2*(h+2);
    float *inp3 = inp0+3*(h+2);

    float *dedy_base = v_dedy+z*O_CHANNELS*w*h;
    v_dedx[x+(y+z*h)*w] = dedy_base[x+y*h];
    inp0[y+1]= dedy_base[x+(y+w)*h];
    inp1[y+1]= dedy_base[x+(y+w*2)*h];
    inp2[y+1]= dedy_base[x+(y+w*3)*h];
    inp3[y+1]= dedy_base[x+(y+w*4)*h];
    inp0[0] = 0.f; inp0[h+1]=0.f;
    inp1[0] = 0.f; inp1[h+1]=0.f;
    inp2[0] = 0.f; inp2[h+1]=0.f;
    inp3[0] = 0.f; inp3[h+1]=0.f;
    __syncthreads();

    float *tmp_base = v_tmp+z*h*w*2;
    tmp_base[x+y*w] = inp0[y]-inp0[y+2]+inp1[y+2]+inp1[y+1]*2+inp1[y];
    tmp_base[x+(y+h)*w] = inp2[y]-inp2[y+2]+inp3[y+2]+inp3[y+1]*2+inp3[y];
}

void __global__ sconv_packed_bwd_row_f32_ker(
        float *v_dedx, float *v_tmp, int w, int h, int bsize) {
    const int x = threadIdx.x;
    const int y = threadIdx.y + blockIdx.y*blockDim.y;
    const int z = threadIdx.z + blockIdx.z*blockDim.z;
    const int by = blockDim.y;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;

    if(x>=w) return;
    if(y>=h) return;
    if(z>=bsize) return;

    extern __shared__ float _inp[];

    float *tmp_base = v_tmp+z*w*h*2;
    float *inp0 = _inp + (ty+tz*by)*2*(w+2);
    float *inp1 = inp0 + w+2;
    inp0[x+1] = tmp_base[x+y*w];
    inp1[x+1] = tmp_base[x+(y+h)*w];
    inp0[0] = 0.f;
    inp0[w+1] = 0.f;
    inp1[0] = 0.f;
    inp1[w+1] = 0.f;
    __syncthreads();

    v_dedx[x+(y+z*h)*w] +=
        inp0[x]-inp0[x+2]+inp1[x+2]+inp1[x+1]*2+inp1[x];
}
