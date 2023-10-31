typedef opaque mem_data<>;

typedef unsigned hyper size_t;
typedef unsigned hyper ptr;
typedef hyper ll;
typedef opaque rpc_cuda_device_prop[1032];
typedef opaque rpc_matmul_heuristic_result[96];

struct dint {
    int i1;
    int i2;
};

struct dsz {
    size_t sz1;
    size_t sz2;
};

struct ptrsz {
    ptr p;
    size_t s;
};

struct matmul_hr {
    rpc_matmul_heuristic_result p;
    int s;
};

struct cuda_channel_format_desc {
    int f;
    int w;
    int x;
    int y;
    int z;
};

struct pitche_ptr {
    size_t pitch;
    ptr ptr;
    size_t xsize;
    size_t ysize;
};

struct rpc_dim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
};

struct int2d1 {
    int i[2];
    double d;
};

struct int1d3 {
    int i;
    double d[3];
};

union cudnn_scaling_t switch (int dataType) {
case 2:
case 0:
    float f;
case 1:
    double d;
};

union int_result switch (int err) {
case 0:
    int data;
default:
    void;
};

union dint_result switch (int err) {
case 0:
    dint data;
default:
    void;
};

union float_result switch (int err) {
case 0:
    float data;
default:
    void;
};

union d_result switch (int err) {
case 0:
    double data;
default:
    void;
};

union u64_result switch (int err) {
case 0:
    unsigned hyper u64;
default:
    void;
};

union dsz_result switch (int err) {
case 0:
    dsz data;
default:
    void;
};

union sz_result switch (int err) {
case 0:
    size_t data;
default:
    void;
};

union ptr_result switch (int err) {
case 0:
    ptr ptr;
default:
    void;
};

union pptr_result switch (int err) {
case 0:
    pitche_ptr ptr;
default:
    void;
};

union str_result switch (int err) {
case 0:
    string str<128>;
default:
    void;
};

union ptrsz_result switch (int err) {
case 0:
    ptrsz data;
default:
    void;
};

union matmul_hr_result switch (int err) {
case 0:
    matmul_hr data;
default:
    void;
};

/* memory allocated for RPC. */
/* Freed rpc_cd_prog_1_freeresult by after RPC. */
union mem_result switch (int err) {
case 0:
    mem_data data;
default:
    void;
};

union cuda_device_prop_result switch (int err) {
case 0:
    rpc_cuda_device_prop data;
default:
    void;
};

union int3_result switch (int err) {
case 0:
    int data[3];
default:
    void;
};

union int4_result switch (int err) {
case 0:
    int data[4];
default:
    void;
};

union int5_result switch (int err) {
case 0:
    int data[5];
default:
    void;
};

union int6_result switch (int err) {
case 0:
    int data[6];
default:
    void;
};

union int8_result switch (int err) {
case 0:
    int data[8];
default:
    void;
};

union int9_result switch (int err) {
case 0:
    int data[9];
default:
    void;
};

union int2d1_result switch (int err) {
case 0:
    int2d1 data;
default:
    void;
};

union int1d3_result switch (int err) {
case 0:
    int1d3 data;
default:
    void;
};

program RPC_CD_PROG {
    version RPC_CD_VERS {
        int          rpc_checkpoint(void)                                         = 0;
        int          rpc_deinit(void)                                             = 1;
        int          rpc_printmessage(string)                                     = 2;
        int          rpc_dlopen(string)                                           = 3;
        ptr_result   rpc_register_function(ptr, ptr, string, string, int)        = 50;
        int          rpc_elf_load(mem_data, ptr)                                 = 51;
        int          rpc_elf_unload(ptr)                                         = 52;
        int          rpc_register_var(ptr, ptr, ptr, string, int, size_t, int, int) = 53;

        /* RUNTIME API */
        /* ### Device Management ### */
        int_result   CUDA_CHOOSE_DEVICE(mem_data)                               = 101;
        int_result   CUDA_DEVICE_GET_ATTRIBUTE(int, int)                        = 102;
        int_result   CUDA_DEVICE_GET_BY_PCI_BUS_ID(string<>)                    = 103;
        int_result   CUDA_DEVICE_GET_CACHE_CONFIG(void)                         = 104;
        u64_result   CUDA_DEVICE_GET_LIMIT(int)                                 = 105;
        /*mem_result CUDA_DEVICE_GET_NVSCISYNC_ATTRIBUTES(int ,int)             = 106;*/
        int_result   CUDA_DEVICE_GET_P2P_ATTRIBUTE(int, int, int)               = 107;
        str_result   CUDA_DEVICE_GET_PCI_BUS_ID(int, int)                       = 108;
        int_result   CUDA_DEVICE_GET_SHARED_MEM_CONFIG(void)                    = 109;
        dint_result  CUDA_DEVICE_GET_STREAM_PRIORITY_RANGE(void)                = 110;
        u64_result   CUDA_DEVICE_GET_TEXTURE_LMW(cuda_channel_format_desc, int) = 111;
        int          CUDA_DEVICE_RESET(void)                                    = 112;
        int          CUDA_DEVICE_SET_CACHE_CONFIG(int)                          = 113;
        int          CUDA_DEVICE_SET_LIMIT(int, size_t)                         = 114;
        int          CUDA_DEVICE_SET_SHARED_MEM_CONFIG(int)                     = 115;
        int          CUDA_DEVICE_SYNCHRONIZE(void)                              = 116;
        int_result   CUDA_GET_DEVICE(void)                                      = 117;
        int_result   CUDA_GET_DEVICE_COUNT(void)                                = 118;
        int_result   CUDA_GET_DEVICE_FLAGS(void)                                = 119;
        cuda_device_prop_result CUDA_GET_DEVICE_PROPERTIES(int)                 = 120;
        /*int        CUDA_IPC_CLOSE_MEM_HANDLE(ptr)                             = 121;*/
        /*ptr_result CUDA_IPC_GET_EVENT_HANDLE(int)                             = 122;*/
        /*ptr_result CUDA_IPC_GET_MEM_HANDLE(ptr)                               = 123;*/
        /*ptr_result CUDA_IPC_OPEN_EVENT_HANDLE(ptr)                            = 124;*/
        /*ptr_result CUDA_IPC_OPEN_MEM_HANDLE(ptr, int)                         = 125;*/
        int          CUDA_SET_DEVICE(int)                                       = 126;
        int          CUDA_SET_DEVICE_FLAGS(int)                                 = 127;
        int          CUDA_SET_VALID_DEVICES(mem_data, int)                      = 128;

        /* ### Error Handling ### */
        str_result   CUDA_GET_ERROR_NAME(int)                                   = 140;
        str_result   CUDA_GET_ERROR_STRING(int)                                 = 141;
        int          CUDA_GET_LAST_ERROR(void)                                  = 142;
        int          CUDA_PEEK_AT_LAST_ERROR(void)                              = 143;

        /* ### Stream Management ### */
        int          CUDA_CTX_RESET_PERSISTING_L2CACHE(void)                    = 250;
        /*int        CUDA_STREAM_ADD_CALLBACK(ptr, ptr, mem_data, int)          = 251;*/
        /*int        CUDA_STREAM_ATTACH_MEM_ASYNC(ptr, ptr, size_t, int)        = 252;*/
        /*int        CUDA_STREAM_BEGIN_CAPTURE(ptr, int)                        = 253;*/
        int          CUDA_STREAM_COPY_ATTRIBUTES(ptr, ptr)                      = 254;
        ptr_result   CUDA_STREAM_CREATE(void)                                   = 255;
        ptr_result   CUDA_STREAM_CREATE_WITH_FLAGS(int)                         = 256;
        ptr_result   CUDA_STREAM_CREATE_WITH_PRIORITY(int, int)                 = 257;
        int          CUDA_STREAM_DESTROY(ptr)                                   = 258;
        /*ptr_result CUDA_STREAM_END_CAPTURE(ptr)                               = 259;*/
        /* ?         CUDA_STREAM_GET_ATTRIBUTE(ptr, int)                        = 260;*/
        /* ?         CUDA_STREAM_GET_CAPTURE_INFO(ptr)                          = 261;*/
        int_result   CUDA_STREAM_GET_FLAGS(ptr)                                 = 262;
        int_result   CUDA_STREAM_GET_PRIORITY(ptr)                              = 263;
        int_result   CUDA_STREAM_IS_CAPTURING(ptr)                              = 264;
        int          CUDA_STREAM_QUERY(ptr)                                     = 265;
        /*int        CUDA_STREAM_SET_ATTRIBUTE(ptr, int, ?)                     = 266;*/
        int          CUDA_STREAM_SYNCHRONIZE(ptr)                               = 267;
        int          CUDA_STREAM_WAIT_EVENT(ptr, ptr, int)                      = 268;
        int_result   CUDA_THREAD_EXCHANGE_STREAM_CAPTURE_MODE(int)              = 269;

        /* ### Event Management ### */
        ptr_result   CUDA_EVENT_CREATE(void)                                    = 280;
        ptr_result   CUDA_EVENT_CREATE_WITH_FLAGS(int)                          = 281;
        int          CUDA_EVENT_DESTROY(ptr)                                    = 282;
        float_result CUDA_EVENT_ELAPSED_TIME(ptr, ptr)                          = 283;
        int          CUDA_EVENT_QUERY(ptr)                                      = 284;
        int          CUDA_EVENT_RECORD(ptr, ptr)                                = 285;
        int          CUDA_EVENT_RECORD_WITH_FLAGS(ptr, ptr, int)                = 286;
        int          CUDA_EVENT_SYNCHRONIZE(ptr)                                = 287;

        /* ### External Resource Interoperability ### */
        /* NOT IMPLEMENTED */

        /* ### Execution Control ### */
        mem_result   CUDA_FUNC_GET_ATTRIBUTES(ptr)                              = 310;
        int          CUDA_FUNC_SET_ATTRIBUTES(ptr, int, int)                    = 311;
        int          CUDA_FUNC_SET_CACHE_CONFIG(ptr, int)                       = 312;
        int          CUDA_FUNC_SET_SHARED_MEM_CONFIG(ptr, int)                  = 313;
        int          CUDA_LAUNCH_COOPERATIVE_KERNEL(ptr, rpc_dim3, 
                          rpc_dim3, mem_data, size_t, ptr)                      = 314;
        /*int        CUDA_LAUNCH_HOST_FUNC(ptr, ptr, mem_data)                  = 316;*/
        int          CUDA_LAUNCH_KERNEL(ptr, rpc_dim3, rpc_dim3,
                          mem_data, size_t, ptr)                                = 317;
        /*d_result   CUDA_SET_DOUBLE_FOR_DEVICE(double)                         = 318;*/
        /*d_result   CUDA_SET_DOUBLE_FOR_HOST(double)                           = 319;*/

        /* ### Occupancy ### */
        u64_result   CUDA_OCCUPANCY_AVAILABLE_DSMPB(ptr, int, int)              = 330;
        int_result   CUDA_OCCUPANCY_MAX_ACTIVE_BPM(ptr, int, size_t)            = 331;
        int_result   CUDA_OCCUPANCY_MAX_ACTIVE_BPM_WITH_FLAGS(ptr, int,
                          size_t, int)                                          = 332;

        /* ### Memory Management ### */
        mem_result   CUDA_ARRAY_GET_INFO(ptr)                                   = 400; 
        mem_result   CUDA_ARRAY_GET_SPARSE_PROPERTIES(ptr)                      = 401;
        int          CUDA_FREE(ptr)                                             = 402;
        int          CUDA_FREE_ARRAY(ptr)                                       = 403;
        int          CUDA_FREE_HOST(int)                                        = 404;   
        /*int        CUDA_FREE_MIPMAPPED_ARRAY(ptr)                             = 405;*/
        /*ptr_result CUDA_GET_MIPMAPPED_ARRAY_LEVEL(ptr, int)                   = 406;*/
        ptr_result   CUDA_GET_SYMBOL_ADDRESS(ptr)                               = 407;
        u64_result   CUDA_GET_SYMBOL_SIZE(ptr)                                  = 408;
        sz_result    CUDA_HOST_ALLOC(size_t, unsigned int)                 = 409;
        int          CUDA_HOST_ALLOC_REGSHM(size_t, ptr)                        = 477;
        ptr_result   CUDA_HOST_GET_DEVICE_POINTER(ptr, int)                     = 410;
        int_result   CUDA_HOST_GET_FLAGS(ptr)                                   = 411;
        /*int        CUDA_HOST_REGISTER(ptr, size_t, int)                       = 412;*/
        /*int        CUDA_HOST_UNREGISTER(ptr)                                  = 413;*/
        ptr_result   CUDA_MALLOC(size_t)                                        = 414;
        pptr_result  CUDA_MALLOC_3D(size_t, size_t, size_t)                     = 415;
        ptr_result   CUDA_MALLOC_3D_ARRAY(cuda_channel_format_desc, 
                          size_t, size_t, size_t, int)                          = 416;
        ptr_result   CUDA_MALLOC_ARRAY(cuda_channel_format_desc,
                          size_t, size_t, int)                                  = 417;
        /*ptr_result CUDA_MALLOC_HOST(ptr, size_t)                              = 418;*/
        /*ptr_result CUDA_MALLOC_MANAGED(size_t, int)                           = 419;*/
        /*ptr_result CUDA_MALLOC_MIPMAPPED_ARRAY(mem_data, size_t, size_t,
                            size_t, int, int)                                   = 420;*/
        ptrsz_result CUDA_MALLOC_PITCH(size_t, size_t)                          = 421;
        int          CUDA_MEM_ADVISE(ptr, size_t, int, int)                     = 422;
        dsz_result   CUDA_MEM_GET_INFO(void)                                    = 423;
        int          CUDA_MEM_PREFETCH_ASYNC(ptr, size_t, int, ptr)             = 424;
        /*mem_result CUDA_MEM_RANGE_GET_ATTRIBUTE(int, ptr, size_t)             = 425;*/
        /*mem_result CUDA_MEM_RANGE_GET_ATTRIBUTES(mem_data, size_t, 
                          ptr, size_t)                                          = 426;*/

        /* ### CUDA_MEMCPY Family ### */
        int          CUDA_MEMCPY_HTOD(ptr, mem_data, size_t)                    = 440;
        mem_result   CUDA_MEMCPY_DTOH(ptr, size_t)                              = 441;
        int          CUDA_MEMCPY_SHM(int, ptr, size_t, int)                     = 442;
        int          CUDA_MEMCPY_DTOD(ptr, ptr, size_t)                         = 443;
        /*mem_result CUDA_MEMCPY_FROM_SYMBOL(ptr, size_t, size_t)               = 444;*/
        int          CUDA_MEMCPY_TO_SYMBOL(ptr, mem_data, size_t, size_t)       = 445;
        int          CUDA_MEMCPY_TO_SYMBOL_SHM(int, ptr, size_t, size_t, int)   = 446;
        int          CUDA_MEMCPY_IB(int, ptr, size_t, int)                      = 447;
        /*int        CUDA_MEMCPY_PEER(ptr, int, ptr, int, size_t)               = 448;*/
        dint_result   CUDA_MEMCPY_MT_HTOD(ptr, size_t, int)                     = 449;
        dint_result   CUDA_MEMCPY_MT_DTOH(ptr, size_t, int)                     = 450;
        int          CUDA_MEMCPY_MT_SYNC(int)                                   = 451;
        int          CUDA_MEMSET(ptr, int, size_t)                              = 470;
        int          CUDA_MEMSET_2D(ptr, size_t, int, size_t, size_t)           = 471;
        int          CUDA_MEMSET_2D_ASYNC(ptr, size_t,
                         int, size_t, size_t, ptr)                              = 472;
        int          CUDA_MEMSET_3D(size_t, ptr, size_t, size_t, int, size_t,
                         size_t, size_t)                                        = 473;
        int          CUDA_MEMSET_3D_ASYNC(size_t, ptr, size_t, size_t, int, 
                         size_t, size_t, size_t, ptr)                           = 474;
        int          CUDA_MEMSET_ASYNC(ptr, int, size_t, ptr)                   = 475;
        /*?          CUDA_MIPMAPPED_ARRAY_GET_SPARSE_PROPERTIES(ptr)            = 476;*/
        /* make_ APIs can be copied on the client side */

        /* ### Unified Addressing ### */
        /* mem_result CUDA_POINTER_GET_ATTRIBUTES(ptr)                          = 490;*/

        /* ### Peer Device Memory Access ### */
        int_result   CUDA_DEVICE_CAN_ACCESS_PEER(int, int)                      = 500;
        int          CUDA_DEVICE_DISABLE_PEER_ACCESS(int)                       = 501;
        int          CUDA_DEVICE_ENABLE_PEER_ACCESS(int, int)                   = 502;

        /* ### Interoperability APIs ### */
        /* NOT IMPLEMENTED */

        /* ### Texture Object Management ### */
        /* NOT IMPLEMENTED */

        /* ### Surface Object Management ### */
        /* NOT IMPLEMENTED */

        /* ### Version Management ### */
        int_result   CUDA_DRIVER_GET_VERSION(void)                              = 510;
        int_result   CUDA_RUNTIME_GET_VERSION(void)                             = 511;

        /* ### Graph Management ### */
        /* NOT IMPLEMENTED */

        /* ### Profiler Control ### */
        int          CUDA_PROFILER_START(void)                                  = 701;
        int          CUDA_PROFILER_STOP(void)                                   = 702;
        /* NOT IMPLEMENTED */

        /* DRIVER API */
        int_result   rpc_cuDeviceGetCount(void)                                = 1002;
        int          rpc_cuInit(int)                                           = 1003;
        int_result   rpc_cuDriverGetVersion(void)                              = 1004;
        int_result   rpc_cuDeviceGet(int)                                      = 1005;
        str_result   rpc_cuDeviceGetName(int)                                  = 1006;
        u64_result   rpc_cuDeviceTotalMem(int)                                 = 1007;
        int_result   rpc_cuDeviceGetAttribute(int, int)                        = 1008;
        str_result   rpc_cuDeviceGetUuid(int)                                  = 1009;
        ptr_result   rpc_cuCtxGetCurrent(void)                                 = 1010;
        int          rpc_cuCtxSetCurrent(ptr)                                  = 1011;
        ptr_result   rpc_cuDevicePrimaryCtxRetain(int)                         = 1012;
        ptr_result   rpc_cuModuleGetFunction(ptr, string)                      = 1013;
/*        ptr_result   rpc_cuGetExportTable(string)                              = 1014;*/
        ptr_result   rpc_cuMemAlloc(ptr)                                       = 1015;
        int_result   rpc_cuCtxGetDevice(void)                                  = 1016;
        int          rpc_cuMemcpyHtoD(ptr, mem_data)                           = 1017;
        int          rpc_cuLaunchKernel(ptr, unsigned int, unsigned int,
                         unsigned int, unsigned int, unsigned int,
                         unsigned int, unsigned int, ptr, mem_data)            = 1018;
        ptr_result   rpc_cuModuleLoad(string<>)                                = 1019;
        str_result   rpc_cuGetErrorString(int)                                 = 1020;
        int          rpc_cuModuleUnload(ptr)                                   = 1021;
        dint_result  rpc_cuDevicePrimaryCtxGetState(int)                       = 1022;
        mem_result   rpc_cuDeviceGetProperties(int)                            = 1023;
        dint_result  rpc_cuDeviceComputeCapability(int)                        = 1024;
        int_result   rpc_cuDeviceGetP2PAttribute(int, ptr, ptr)                = 1025; 
        ptr_result   rpc_cuModuleLoadData(mem_data mem)                        = 1026;

        /* HIDDEN DRIVER API */
/*        ptr_result   rpc_hidden_get_device_ctx(int)                            = 1101;
        ptr_result   rpc_hidden_get_module(ptr, ptr, ptr, int)                 = 1105;
        ptr_result   rpc_hidden_1_1(void)                                      = 1111;
        void         rpc_hidden_1_3(ptr, ptr)                                  = 1113;
        ptr_result   rpc_hidden_1_5(void)                                      = 1115;
        void         rpc_hidden_2_1(ptr)                                       = 1121;
        int          rpc_hidden_3_0(int, ptr, ptr)                             = 1130;
        mem_result   rpc_hidden_3_2(int, ptr)                                  = 1132;*/

        /* CUSOLVER */
        ptr_result   rpc_cusolverDnCreate(void)                                = 2001;
        int          rpc_cusolverDnSetStream(ptr, ptr)                         = 2002;
        int_result   rpc_cusolverDnDgetrf_bufferSize(ptr, int, int, ptr, int)  = 2003;
        int          rpc_cusolverDnDgetrf(ptr, int, int, ptr,
                         int, ptr, ptr, ptr)                                   = 2004;
        int          rpc_cusolverDnDgetrs(ptr, int, int, int, ptr,
                         int, ptr, ptr, int, ptr)                              = 2005;
        int          rpc_cusolverDnDestroy(ptr)                                = 2006;

        /* CUBLAS */
        ptr_result   rpc_cublasCreate(void)                                    = 3001;
        int          rpc_cublasDgemm(ptr, int, int, int, int, int, double,
                         ptr, int, ptr, int, double, ptr, int)                 = 3002;
        int          rpc_cublasDestroy(ptr)                                    = 3003;
        int          rpc_cublasSgemm(ptr, int, int, int, int, int, float,
                         ptr, int, ptr, int, float, ptr, int)                 = 3004;
        int          rpc_cublasSgemv(ptr, int, int, int, float,
                         ptr, int, ptr, int, float, ptr, int)                 = 3005;
        int          rpc_cublasDgemv(ptr, int, int, int, double,
                         ptr, int, ptr, int, double, ptr, int)                 = 3006;
        int          rpc_cublasSgemmEx(ptr, int, int, int, int, int, float,
                         ptr, int, int, ptr, int, int, float, ptr, int, int)                 = 3007;
        int          rpc_cublasSetStream(ptr handle, ptr streamId)                             = 3008;
        int          rpc_cublasSetWorkspace(ptr handle, ptr workspace, size_t workspaceSizeInBytes) = 3009;
        int          rpc_cublasSetMathMode(ptr handle, int mode) = 3010;

        /* NVML */
        int_result   rpc_nvmlDeviceGetCount_v2(void)                           = 4000;
        int          rpc_nvmlInitWithFlags(int)                                = 4001;
        int          rpc_nvmlInit_v2(void)                                     = 4002;
        int          rpc_nvmlShutdown(void)                                    = 4003;
        
        /* CUDNN */
        size_t      rpc_cudnnGetVersion(void) = 5000;
        size_t      rpc_cudnnGetMaxDeviceVersion(void) = 5001;
        size_t      rpc_cudnnGetCudartVersion(void) = 5002;
        string      rpc_cudnnGetErrorString (int status) = 5003;
        int_result  rpc_cudnnQueryRuntimeError(ptr handle, int mode) = 5004;
        int_result  rpc_cudnnGetProperty(int type) = 5005;
        ptr_result  rpc_cudnnCreate(void) = 5006;
        int         rpc_cudnnDestroy(ptr handle) = 5007;
        int         rpc_cudnnSetStream(ptr handle, ptr streamId) = 5008;
        ptr_result  rpc_cudnnGetStream(ptr handle) = 5009;
        ptr_result  rpc_cudnnCreateTensorDescriptor(void) = 5010;
        int         rpc_cudnnSetTensor4dDescriptor(ptr tensorDesc, int format, int dataType, int n, int c, int h, int w) = 5011;
        int         rpc_cudnnSetTensor4dDescriptorEx(ptr tensorDesc, int dataType, int n, int c, int h, int w, int nStride, int cStride, int hStride, int wStride) = 5012;
        int9_result rpc_cudnnGetTensor4dDescriptor(ptr tensorDesc) = 5013;
        int         rpc_cudnnSetTensorNdDescriptor(ptr tensorDesc, int dataType, int nbDims, mem_data dimA, mem_data strideA) = 5014;
        int         rpc_cudnnSetTensorNdDescriptorEx(ptr tensorDesc, int format, int dataType, int nbDims, mem_data dimA) = 5015;
        mem_result  rpc_cudnnGetTensorNdDescriptor(ptr tensorDesc, int nbDimsRequested) = 5016;
        sz_result   rpc_cudnnGetTensorSizeInBytes(ptr tensorDesc) = 5017;
        int         rpc_cudnnDestroyTensorDescriptor(ptr tensorDesc) = 5018;
        /*
        sz_result   rpc_cudnnInitTransformDest(ptr transformDesc, ptr srcDesc, ptr destDesc) = 5019;
        ptr_result  rpc_cudnnCreateTensorTransformDescriptor(void) = 5020;
        int         rpc_cudnnSetTensorTransformDescriptor(ptr transformDesc, uint32_t nbDims, int destFormat, mem_data padBeforeA, mem_data padAfterA, mem_data foldA, int direction) = 5021;
        mem_result  rpc_cudnnGetTensorTransformDescriptor(ptr transformDesc, uint32_t nbDimsRequested) = 5022;
        int         rpc_cudnnDestroyTensorTransformDescriptor(ptr transformDesc) = 5023;
        */
        int         rpc_cudnnTransformTensor(ptr handle, cudnn_scaling_t alpha, ptr xDesc, ptr x, cudnn_scaling_t beta, ptr yDesc, ptr y) = 5024;
        /*
        ptr_result  rpc_cudnnTransformTensorEx(ptr handle, ptr transDesc, cudnn_scaling_t alpha, ptr srcDesc, cudnn_scaling_t srcData, cudnn_scaling_t beta, ptr destDesc) = 5025;
        */
        int  rpc_cudnnAddTensor(ptr handle, cudnn_scaling_t alpha, ptr aDesc, ptr A, cudnn_scaling_t beta, ptr cDesc, ptr C) = 5026;
        /*
        ptr_result  rpc_cudnnCreateOpTensorDescriptor(void) = 5027;
        int         rpc_cudnnSetOpTensorDescriptor(ptr opTensorDesc, int opTensorOp, int opTensorCompType, int opTensorNanOpt) = 5028;
        int3_result rpc_cudnnGetOpTensorDescriptor(ptr opTensorDesc) = 5029;
        int         rpc_cudnnDestroyOpTensorDescriptor(ptr opTensorDesc) = 5030;
        mem_result  rpc_cudnnOpTensor(ptr handle, ptr opTensorDesc, cudnn_scaling_t alpha1, ptr aDesc, mem_data A, cudnn_scaling_t alpha2, ptr bDesc, mem_data B, cudnn_scaling_t beta, ptr  cDesc) = 5031;
        ptr_result  rpc_cudnnCreateReduceTensorDescriptor(void) = 5032;
        int         rpc_cudnnSetReduceTensorDescriptor(ptr reduceTensorDesc, int reduceTensorOp, int reduceTensorCompType, int reduceTensorNanOpt, int reduceTensorIndices, int reduceTensorIndicesType) = 5033;
        int5_result rpc_cudnnGetReduceTensorDescriptor(ptr reduceTensorDesc) = 5034;
        int         rpc_cudnnDestroyReduceTensorDescriptor(ptr reduceTensorDesc) = 5035;
        sz_result   rpc_cudnnGetReductionIndicesSize(ptr handle, ptr reduceTensorDesc, ptr aDesc, ptr cDesc) = 5036;
        sz_result   rpc_cudnnGetReductionWorkspaceSize(ptr handle, ptr reduceTensorDesc, ptr aDesc, ptr cDesc) = 5037;
        mem_result  rpc_cudnnReduceTensor(ptr handle, ptr reduceTensorDesc, ptr indices, size_t indicesSizeInBytes, ptr workspace, size_t workspaceSizeInBytes, cudnn_scaling_t alpha, ptr aDesc, ptr A, cudnn_scaling_t beta, ptr cDesc, ptr C) = 5038;
        int         rpc_cudnnSetTensor(ptr handle, ptr yDesc, ptr y, mem_data valuePtr) = 5039;
        int         rpc_cudnnScaleTensor(ptr handle, ptr yDesc, ptr y, cudnn_scaling_t alpha) = 5040; */
        
        ptr_result  rpc_cudnnCreateFilterDescriptor(void) = 5041;
        int         rpc_cudnnSetFilter4dDescriptor(ptr filterDesc, int dataType, int format, int k, int c, int h, int w) = 5042;
        int6_result rpc_cudnnGetFilter4dDescriptor(ptr filterDesc) = 5043;
        int         rpc_cudnnSetFilterNdDescriptor(ptr filterDesc, int dataType, int format, int nbDims, mem_data filterDimA) = 5044;
        mem_result  rpc_cudnnGetFilterNdDescriptor(ptr filterDesc, int nbDimsRequested) = 5045;
        sz_result   rpc_cudnnGetFilterSizeInBytes(ptr filterDesc) = 5046;
        int         rpc_cudnnTransformFilter(ptr handle, ptr transDesc, cudnn_scaling_t alpha, ptr srcDesc, ptr srcData, cudnn_scaling_t beta, ptr destDesc, ptr destData) = 5047;
        int         rpc_cudnnDestroyFilterDescriptor(ptr filterDesc) = 5048;
        int         rpc_cudnnSoftmaxForward(ptr handle, int algo, int mode, cudnn_scaling_t alpha, ptr xDesc, ptr x, cudnn_scaling_t beta, ptr yDesc, ptr y) = 5049;
        ptr_result  rpc_cudnnCreatePoolingDescriptor(void) = 5050;
        int         rpc_cudnnSetPooling2dDescriptor(ptr poolingDesc, int mode, int maxpoolingNanOpt, int windowHeight, int windowWidth, int verticalPadding, int horizontalPadding, int verticalStride, int horizontalStride) = 5051;
        int8_result rpc_cudnnGetPooling2dDescriptor(ptr poolingDesc) = 5052;
        int         rpc_cudnnSetPoolingNdDescriptor(ptr poolingDesc, int mode, int maxpoolingNanOpt, int nbDims, mem_data windowDimA, mem_data paddingA, mem_data strideA) = 5053;
        mem_result  rpc_cudnnGetPoolingNdDescriptor(ptr poolingDesc, int nbDimsRequested) = 5054;
        mem_result  rpc_cudnnGetPoolingNdForwardOutputDim(ptr poolingDesc, ptr inputTensorDesc, int nbDims) = 5055;
        int4_result rpc_cudnnGetPooling2dForwardOutputDim(ptr poolingDesc, ptr inputTensorDesc) = 5056;
        int         rpc_cudnnDestroyPoolingDescriptor(ptr poolingDesc) = 5057;
        int         rpc_cudnnPoolingForward(ptr handle, ptr poolingDesc, cudnn_scaling_t alpha, ptr xDesc, ptr x, cudnn_scaling_t beta, ptr yDesc, ptr y) = 5058;
        ptr_result  rpc_cudnnCreateActivationDescriptor(void) = 5059;
        int         rpc_cudnnSetActivationDescriptor(ptr activationDesc, int mode, int reluNanOpt, double coef) = 5060;
        int2d1_result rpc_cudnnGetActivationDescriptor(ptr activationDesc) = 5061;
        int         rpc_cudnnSetActivationDescriptorSwishBeta(ptr activationDesc, double swish_beta) = 5062;
        d_result    rpc_cudnnGetActivationDescriptorSwishBeta(ptr activationDesc) = 5063;
        int         rpc_cudnnDestroyActivationDescriptor(ptr activationDesc) = 5064;
        int         rpc_cudnnActivationForward(ptr handle, ptr activationDesc, cudnn_scaling_t alpha, ptr xDesc, ptr x, cudnn_scaling_t beta, ptr yDesc, ptr y) = 5065;
        ptr_result  rpc_cudnnCreateLRNDescriptor(void) = 5066;
        int         rpc_cudnnSetLRNDescriptor(ptr normDesc, unsigned lrnN, double lrnAlpha, double lrnBeta, double lrnK) = 5067;
        int1d3_result rpc_cudnnGetLRNDescriptor(ptr normDesc) = 5068;
        int         rpc_cudnnDestroyLRNDescriptor(ptr lrnDesc) = 5069;
        int         rpc_cudnnLRNCrossChannelForward(ptr handle, ptr normDesc, int lrnMode, cudnn_scaling_t alpha, ptr xDesc, ptr x, cudnn_scaling_t beta, ptr yDesc, ptr y) = 5070;
        /* cudnn cnn inference */
        ptr_result  rpc_cudnnCreateConvolutionDescriptor(void) = 5301;
        int         rpc_cudnnDestroyConvolutionDescriptor(ptr convDesc) = 5302;
        mem_result  rpc_cudnnGetConvolutionNdForwardOutputDim(ptr convDesc, ptr inputTensorDesc, ptr filterDesc, int nbDims) = 5303;
        int         rpc_cudnnSetConvolutionNdDescriptor(ptr convDesc, int arrayLength, mem_data padA,  mem_data filterStrideA, mem_data dilationA,  int mode,  int computeType) = 5304;
        mem_result rpc_cudnnGetConvolutionForwardAlgorithm_v7(ptr handle, ptr srcDesc, ptr filterDesc, ptr convDesc, ptr destDesc, int requestedAlgoCount) = 5305;
        mem_result rpc_cudnnFindConvolutionForwardAlgorithm(ptr handle, ptr xDesc, ptr wDesc, ptr convDesc, ptr yDesc, int requestedAlgoCount) = 5306;
        sz_result rpc_cudnnGetConvolutionForwardWorkspaceSize(ptr handle, ptr xDesc, ptr wDesc, ptr convDesc, ptr yDesc, int algo) = 5307;
        int rpc_cudnnConvolutionForward(ptr handle, cudnn_scaling_t alpha, ptr xDesc, ptr x, ptr wDesc, ptr w, ptr convDesc, int algo, ptr workSpace, size_t workSpaceSizeInBytes, cudnn_scaling_t beta, ptr yDesc, ptr y) = 5308;
        ptr_result rpc_cudnnBackendCreateDescriptor(int descriptorType) = 5309;
        int rpc_cudnnBackendDestroyDescriptor(ptr descriptor) = 5310;
        int rpc_cudnnBackendInitialize(ptr descriptor) = 5311;
        int rpc_cudnnBackendFinalize(ptr descriptor) = 5312;
        int rpc_cudnnBackendSetAttribute(ptr descriptor,
                         int attributeName,
                         int attributeType,
                         hyper elementCount,
                         mem_data arrayOfElements) = 5313;
        mem_result rpc_cudnnBackendGetAttribute(ptr descriptor,
                            int attributeName,
                            int attributeType,
                            hyper requestedElementCount) = 5314;
        int rpc_cudnnBackendExecute(ptr handle, ptr executionPlan, ptr variantPack) = 5315;
        int rpc_cudnnSetConvolutionGroupCount(ptr convDesc, int groupCount) = 5316;
        int rpc_cudnnsetconvolutionmathtype(ptr convDesc, int mathType) = 5317;
        ptr_result rpc_cublasltcreate(void) = 5318;
        ptr_result rpc_cublasltmatmuldesccreate(int computeType, int scaleType) = 5319;
        matmul_hr_result rpc_cublasltmatmulalgogetheuristic(ptr handle, ptr operationDesc, ptr aDesc, ptr bDesc, ptr cDesc, ptr dDesc, ptr preference, int requestedAlgoCount) = 5320;
        int rpc_cublasltmatmuldescsetattribute(ptr matmulDesc, int attr, mem_data data) = 5321;
        int rpc_cublasltmatmuldescdestroy(ptr matmulDesc) = 5322;
        ptr_result rpc_cublasltmatrixlayoutcreate(int type, uint64_t row, uint64_t cols, int64_t ld) = 5323;
        ptr_result rpc_cublasltmatmulpreferencecreate(void) = 5324;
        int rpc_cublasltmatmulpreferencedestroy(ptr pref) = 5325;
        int rpc_cublasltmatrixlayoutdestroy(ptr matLayout) = 5326;
        int rpc_cublasltmatmul(ptr lightHandle,ptr computeDesc,float alpha,ptr A,ptr Adesc,ptr B,ptr Bdesc,float beta,ptr C,ptr Cdesc,ptr D,ptr Ddesc,ptr algo,ptr workspace,size_t workspaceSizeInBytes,ptr stream) = 5327;
        int_result rpc_cublasgetmathmode(ptr handle) = 5328;
        int rpc_cublasgemmstridedbatchedex(ptr handle, int transa, int transb, int m,int n,int k,float alpha,ptr A, int Atype, int lda, ll strideA,ptr B,int Btype,int ldb, ll strideB,float beta,   ptr C,int Ctype,int ldc, ll strideC,int batchCount,int computeType,int algo) = 5329;
        int rpc_cublasgemmex(ptr, int, int, int, int, int, float,ptr, int, int, ptr, int, int, float, ptr, int, int, int, int) = 5330;
        int rpc_cublasgemmstridedbatched(ptr handle, int transa, int transb, int m,int n,int k,float alpha,ptr A, int lda, ll strideA,ptr B,int ldb, ll strideB,float beta,   ptr C,int ldc, ll strideC,int batchCount) = 5331;
    } = 1;
} = 99;
