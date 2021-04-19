/**
 * Kernel function for converting BGR image format to YUV444 format.
 * Output byte order: Y0U0V0 Y1U1V1 Y2U2V2 Y3U3V3 etc.
 * Each output pixel consists of all 3 YUV components.
 *
 * @param d_in          input BGR image
 * @param d_out         output YUV444 image
 * @param imgwidth      image widht
 * @param imgheight     image height
 */
__global__ void bgr2yuv444 (uchar3 *d_in, unsigned char *d_out, unsigned int imgwidth, unsigned int imgheight);

/**
 * Kernel function for converting YUV444 image format to YUV422 format.
 * Output byte order: U0Y0V0 Y1 U2Y2V2 Y3 etc.
 * Even output pixels consist of all 3 YUV components, while odd ones contain only Y component.
 *
 * @param d_in          input YUV444 image
 * @param d_out         output YUV422 image
 * @param img_size      image size i.e. imgwidht * imgheight
 */
__global__ void yuv444toyuv422 (unsigned char *d_in, unsigned char *d_out, unsigned int img_size);

/**
 * Kernel function for extracting each Y, U and V component/channel separately from YUV422 format.
 *
 * @param d_in          input YUV422 image
 * @param y_out         output Y channel
 * @param u_out         output U channel
 * @param v_out         output V channel
 * @param img_size      image size i.e. imgwidht * imgheight
 */
__global__ void extract_channels (unsigned char *d_in, unsigned char *y_out, unsigned char *u_out, unsigned char *v_out,
    unsigned int img_size);

/**
 * Kernel function for converting YUV422 image back to BGR image format.
 *
 * @param d_in          input YUV4222 image
 * @param d_out_bgr     output BGR image
 * @param img_size      size i.e. imgwidht * imgheight
 */
__global__ void yuv422tobgr (unsigned char *d_in, uchar3 *d_out_bgr, unsigned int img_size);

/**
 * Auxiliary device functions for converting B, G and R to Y, U and V components and vice versa.
 */

__device__ unsigned char y_component (unsigned char b, unsigned char g, unsigned char r);

__device__ unsigned char u_component (unsigned char y, unsigned char r, unsigned char delta);

__device__ unsigned char v_component (unsigned char y, unsigned char b, unsigned char delta);

__device__ unsigned char r_component (unsigned char y, unsigned char u, unsigned char delta);

__device__ unsigned char g_component (unsigned char y, unsigned char u, unsigned char v, unsigned char delta);

__device__ unsigned char b_component (unsigned char y, unsigned char v, unsigned char delta);