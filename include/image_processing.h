/**
 * Wrapper function for converting BGR image format to YUV444 format.
 * The function executes kernel for converting BGR image format to YUV444 format.
 * Function returns time kernel took for execution.
 *
 * @param d_in          input BGR image
 * @param d_out         output YUV444 image
 * @param imgwidth      image widht
 * @param imgheight     image height
 *
 * @return gpu_time     time kernel took for execution, in seconds
 */
double bgr_to_yuv444(uchar3 *d_in, unsigned char* d_out_yuv444, unsigned int imgheight, unsigned int imgwidth);

/**
 * Wrapper function for converting YUV444 image format to YUV422 format.
 * The function executes kernel for converting YUV444 image format to YUV422 format.
 * Function returns time kernel took for execution.
 *
 * @param d_out_yuv444  input YUV444 image
 * @param d_out_yuv422  output YUV422 image
 * @param imgwidth      image widht
 * @param imgheight     image height
 *
 * @return gpu_time     time kernel took for execution, in seconds
 */
double yuv444_to_yuv422(unsigned char* d_out_yuv444, unsigned char* d_out_yuv422, unsigned int imgheight, unsigned int imgwidth);

/**
 * Wrapper function for extracting single separate Y, U and Y channels from YUV422 format.
 * The function executes kernel for extracting separate Y, U and V channels.
 * Function returns time kernel took for execution.
 *
 * @param d_out_yuv422  input YUV422 image
 * @param d_y_channel   output Y channel
 * @param d_u_channel   output U channel
 * @param d_v_channel   output V channel
 * @param imgwidth      image widht
 * @param imgheight     image height
 *
 * @return gpu_time     time kernel took for execution, in seconds
 */
double extract_separate_channels(unsigned char* d_out_yuv422, unsigned char* d_y_channel, unsigned char* d_u_channel,
    unsigned char* d_v_channel, unsigned int imgheight, unsigned int imgwidth);

/**
 * Wrapper function for converting YUV422 image format to BGR format.
 * The function executes kernel for converting YUV22 image format back to BGR format.
 * Function returns time kernel took for execution.
 *
 * @param d_out_yuv422  input YUV422 image
 * @param d_out_bgr     output BGR image
 * @param imgwidth      image widht
 * @param imgheight     image height
 *
 * @return gpu_time     time kernel took for execution, in seconds
 */
double yuv422_to_bgr (unsigned char* d_out_yuv422, uchar3 *d_out_bgr, unsigned int imgheight, unsigned int imgwidth);