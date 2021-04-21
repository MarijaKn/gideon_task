double bgr_to_yuv444(uchar3 *d_in, unsigned char* d_out_yuv444, unsigned int imgheight, unsigned int imgwidth) {

    // Set thread and grid size
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((imgwidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (imgheight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    clock_t start, end;
    start = clock();
    bgr2yuv444<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out_yuv444, imgwidth, imgheight);
    cudaDeviceSynchronize();
    end = clock();

    double gpu_time = (double) (end-start) / CLOCKS_PER_SEC;

    return gpu_time;
}

double yuv444_to_yuv422(unsigned char* d_out_yuv444, unsigned char* d_out_yuv422, unsigned int imgheight, unsigned int imgwidth) {

    // Set thread and block size
    unsigned int n_threads = 32;
    unsigned int n_blocks = (unsigned int) ceil((floor) (imgwidth * imgheight) / n_threads);

    clock_t start = clock();
    yuv444toyuv422<<<n_blocks, n_threads>>>(d_out_yuv444, d_out_yuv422, imgwidth*imgheight);
    cudaDeviceSynchronize();
    clock_t end = clock();

    double gpu_time = (double) (end-start) / CLOCKS_PER_SEC;

    return gpu_time;
}

double extract_separate_channels(unsigned char* d_out_yuv422, unsigned char* d_y_channel, unsigned char* d_u_channel,
    unsigned char* d_v_channel, unsigned int imgheight, unsigned int imgwidth) {

    // Set thread and block size
    unsigned int n_threads = 32;
    unsigned int n_blocks = (unsigned int) ceil((floor) (imgwidth * imgheight) / n_threads);

    clock_t start = clock();
    extract_channels<<<n_blocks, n_threads>>>(d_out_yuv422, d_y_channel, d_u_channel, d_v_channel, imgwidth*imgheight);
    cudaDeviceSynchronize();
    clock_t end = clock();

    double gpu_time = (double) (end-start) / CLOCKS_PER_SEC;

    return gpu_time;
}

double yuv422_to_bgr (unsigned char* d_out_yuv422, uchar3 *d_out_bgr, unsigned int imgheight, unsigned int imgwidth) {

    // Set thread and block size
    unsigned int n_threads = 32;
    unsigned int n_blocks = (unsigned int) ceil((floor) (imgwidth * imgheight) / n_threads);

    clock_t start = clock();
    yuv422tobgr<<<n_blocks, n_threads>>>(d_out_yuv422, d_out_bgr, imgheight*imgwidth);
    cudaDeviceSynchronize();
    clock_t end = clock();

    double gpu_time = (double) (end-start) / CLOCKS_PER_SEC;

    return gpu_time;
}