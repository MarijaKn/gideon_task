// NOTE: The conversion formulas I used are those from the task text.
// I can notice that identical formulas are found on wikipedia as well.
// However, after using these formulas in the kernel and then converting using the OpenCV cvrColor function,
// I noticed that my resulting BGR image is a good size, however very blue-ish.
// It seemed to me that I had read one of the BGR components incorrectly.
// Then I noticed that by replacing the B and R channels in the kernel I get the correct BGR image after using cvrColor.

// I think the conversion formulas are correct, and what I assume could be a problem is the OpenCV cvrColor conversion function.
// Finally, I checked my version of OpenCV, and it seems to have some issues:
// https://github.com/opencv/opencv/issues/4946

__device__ unsigned char y_component (unsigned char b, unsigned char g, unsigned char r) {

    unsigned char y = 0.299 * r + 0.587 * g + 0.114 * b;

    return y;
}

__device__ unsigned char u_component (unsigned char y, unsigned char r, unsigned char delta) {

    unsigned char cr = (r - y) * 0.713 + delta;

    return cr;
}

__device__ unsigned char v_component (unsigned char y, unsigned char b, unsigned char delta) {

    unsigned char cb = (b - y) * 0.564 + delta;

    return cb;
}

__device__ unsigned char r_component (unsigned char y, unsigned char u, unsigned char delta) {

    unsigned char r = y + 1.403 * (u - delta);

    return r;
}

__device__ unsigned char g_component (unsigned char y, unsigned char u, unsigned char v, unsigned char delta) {

    unsigned char g = y - 0.414 * (u - delta) - 0.344 * (v - delta);

    return g;
}

__device__ unsigned char b_component (unsigned char y, unsigned char v, unsigned char delta) {

    unsigned char b = y + 1.773 * (v - delta);

    return b;
}

// Output byte order: Y0U0V0 Y1U1V1 Y2U2V2 Y3U3V3 etc.
__global__ void bgr2yuv444 (uchar3 *d_in, unsigned char *d_out, unsigned int imgwidth, unsigned int imgheight) {

    int column_idx = blockIdx.x*blockDim.x+threadIdx.x;
    int row_idx = blockIdx.y*blockDim.y+threadIdx.y;

    if ((row_idx > imgheight) || (column_idx > imgwidth)) {
        return;
    }

    int global_idx =  row_idx * imgwidth + column_idx;

    unsigned char b, g, r = 0;

    // Note: Here is the swapping of B and R component I mentioned at the top of this file
    b = d_in[global_idx].z;
    g = d_in[global_idx].y;
    r = d_in[global_idx].x;

    unsigned char delta = 128;

    // In YUV444 every pixel has each of 3 components
    // Each pixel (global idx) reproduces 3 components on output and they are interspersed with each other

    unsigned char y = y_component(b, g, r);

    d_out[3 * global_idx]       = y;
    d_out[3 * global_idx + 1]   = u_component(y, r, delta);
    d_out[3 * global_idx + 2]   = v_component(y, b, delta);
}


// Output byte order: U0Y0V0 Y1 U2Y2V2 Y3 etc.
__global__ void yuv444toyuv422 (unsigned char *d_in, unsigned char *d_out, unsigned int img_size) {

    int global_idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (global_idx > img_size) {
        return;
    }

    int y_in_idx, y_out_idx;

    // Even threads (even global idx indices i.e. even pixels) consist of all 3 YUV components
    if (global_idx % 2 == 0) {

        y_in_idx = 3 * global_idx;
        int u_in_idx = 3 * global_idx + 1;
        int v_in_idx = 3 * global_idx + 2;

        y_out_idx = 2 * global_idx + 1;
        int u_out_idx = 2 * global_idx;
        int v_out_idx = 2 * global_idx + 2;

        d_out[u_out_idx] = d_in[u_in_idx];
        d_out[y_out_idx] = d_in[y_in_idx];
        d_out[v_out_idx] = d_in[v_in_idx];

    } else {

        // Odd threads (odd global idx indices i.e. odd pixels) consist of only Y component
        y_in_idx  = 3 * global_idx;
        y_out_idx = y_in_idx - (global_idx - 1);

        d_out[y_out_idx] = d_in[y_in_idx];

    }

}

__global__ void extract_channels (unsigned char *d_in, unsigned char *y_out, unsigned char *u_out, unsigned char *v_out,
    unsigned int img_size) {

    int global_idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (global_idx > img_size) {
        return;
    }

    int y_in_idx;

    if (global_idx % 2 == 0) {

        y_in_idx = 2 * global_idx + 1;

        int u_in_idx = 2 * global_idx;
        int v_in_idx = 2 * global_idx + 2;

        int uv_out_idx = u_in_idx / 4;

        u_out[uv_out_idx] = d_in[u_in_idx];
        v_out[uv_out_idx] = d_in[v_in_idx];

    } else {

        y_in_idx = 3 * global_idx - (global_idx - 1);
    }

    y_out[global_idx] = d_in[y_in_idx];
}

__global__ void yuv422tobgr (unsigned char *d_in, uchar3 *d_out_bgr, unsigned int img_size) {

    int global_idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (global_idx > img_size) {
        return;
    }

    unsigned char delta = 128;

    int y_in_idx;
    int y, u, v = 0;

    if (global_idx % 2 == 0) {

        y_in_idx = 2 * global_idx + 1;
        int u_in_idx = 2 * global_idx;
        int v_in_idx = 2 * global_idx + 2;

        y = d_in[y_in_idx];
        u = d_in[u_in_idx];
        v = d_in[v_in_idx];

    } else {

        y_in_idx = 3 * global_idx - (global_idx - 1);

        y = d_in[y_in_idx];
        u = d_in[y_in_idx - 3];
        v = d_in[y_in_idx - 1];
    }

    d_out_bgr[global_idx].z = b_component(y, v, delta);
    d_out_bgr[global_idx].y = g_component(y, u, v, delta);
    d_out_bgr[global_idx].x = r_component(y, u, delta);
}