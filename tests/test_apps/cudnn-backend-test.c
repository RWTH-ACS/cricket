#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


int main(int argc, char** argv) {
    printf("Hello World\n");
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    printf("cudnn created\n");

    cudnnBackendDescriptor_t xDesc;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &xDesc);
    size_t xId = 'x';
    cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID,
                            CUDNN_TYPE_INT64, 1, &xId);
    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_DATA_TYPE,
                            CUDNN_TYPE_DATA_TYPE, 1, &dtype);
    size_t alignment = 4;
    cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                            CUDNN_TYPE_INT64, 1, &alignment);
    size_t tensor_dims = 3;
    cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_DIMENSIONS,
                            CUDNN_TYPE_INT64, 1, &tensor_dims);
    size_t tensor_stride = 1;
    cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_STRIDES,
                            CUDNN_TYPE_INT64, 1, &tensor_stride);
    cudnnBackendFinalize(xDesc);

    cudnnBackendDescriptor_t outDesc;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &outDesc);
    size_t yId = 'y';
    cudnnBackendSetAttribute(outDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID,
                            CUDNN_TYPE_INT64, 1, &id);
    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnBackendSetAttribute(outDesc, CUDNN_ATTR_TENSOR_DATA_TYPE,
                            CUDNN_TYPE_DATA_TYPE, 1, &dtype);
    cudnnBackendSetAttribute(outDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                            CUDNN_TYPE_INT64, 1, &alignment);
    cudnnBackendSetAttribute(outDesc, CUDNN_ATTR_TENSOR_DIMENSIONS,
                            CUDNN_TYPE_INT64, 1, &tensor_dims);
    cudnnBackendSetAttribute(outDesc, CUDNN_ATTR_TENSOR_STRIDES,
                            CUDNN_TYPE_INT64, 1, &tensor_stride);
    cudnnBackendFinalize(outDesc);
    
    printf("cudnn outDesc created\n");

    cudnnBackendDescriptor_t concat;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR, &concat);
    size_t concat_dim = 0;
    cudnnBackendSetAttribute(concat, CUDNN_ATTR_OPERATION_CONCAT_AXIS,
                            CUDNN_TYPE_INT64, 1, &concat_dim);
    cudnnBackendSetAttribute(concat, CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS,
                            CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xDesc);
    cudnnBackendSetAttribute(concat, CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC,
                            CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &outDesc);
    cudnnBackendFinalize(concat);

    printf("cudnn concat created\n");

    cudnnBackendDescriptor_t opset;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &opset);
    cudnnBackendSetAttribute(opset, CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
                            CUDNN_TYPE_HANDLE, 1, &cudnn);
    cudnnBackendSetAttribute(opset, CUDNN_ATTR_OPERATIONGRAPH_OPS,
                            CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &concat);
    cudnnBackendFinalize(opset);

    printf("cudnn opset created\n");

    cudnnBackendDescriptor_t engine;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine);
    cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
                             CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &opset);
    int64_t gidx = 0;
    cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
                             CUDNN_TYPE_INT64, 1, &gidx);
    cudnnBackendFinalize(engine);

    printf("cudnn engine created\n");

    cudnnBackendDescriptor_t engcfg;
    cudnnBackendSetAttribute(engcfg, CUDNN_ATTR_ENGINECFG_ENGINE,
                            CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine);
    cudnnBackendFinalize(engcfg);
    
    printf("cudnn engcfg created\n");

    cudnnBackendDescriptor_t plan;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan);
    cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                            CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engcfg);
    cudnnBackendFinalize(plan);

    printf("cudnn plan created\n");

    int64_t workspaceSize;
    cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                            CUDNN_TYPE_INT64, 1, NULL, &workspaceSize);

    float *xData = (float*)malloc(3 * sizeof(float));
    xData[0] = 1.0f;
    xData[1] = 2.0f;
    xData[2] = 3.0f;
    void *xData_dev = NULL;
    cudaMalloc(&xData_dev, 3 * sizeof(float));
    cudaMemcpy(xData_dev, xData, 3 * sizeof(float), cudaMemcpyHostToDevice);
    void *yData_dev = NULL;
    cudaMalloc(&yData_dev, 3 * sizeof(float));

    void *dev_ptrs[1] = {xData_dev, yData_dev}; // device pointer
    int64_t uids[1] = {'x', 'y'};
    void *workspace = NULL;
    cudaMalloc(&workspace, workspaceSize);

    cudnnBackendDescriptor_t varpack;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &varpack);
    cudnnBackendSetAttribute(varpack, CUDNN_ATTR_1ARIANT_PACK_DATA_POINTERS,
                            CUDNN_TYPE_VOID_PTR, 2, dev_ptrs);
    cudnnBackendSetAttribute(varpack, CUDNN_AT1R_VARIANT_PACK_UNIQUE_IDS,
                            CUDNN_TYPE_INT64, 2, uids);
    cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
                            CUDNN_TYPE_VOID_PTR, 1, &workspace);
    cudnnBackendFinalize(varpack);

    printf("cudnn varpack created\n");
    
    cudnnBackendExecute(cudnn, plan, varpack);

    printf("cudnn executed\n");
    
    cudnnDestroy(cudnn);
    return 0;
}

