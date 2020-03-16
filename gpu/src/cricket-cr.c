#include "cricket-cr.h"

#include <stdio.h>
#include "cuda-tdep.h"

#include <sys/time.h>
#include "cricket-elf.h"
#include "cricket-file.h"
#include "cricket-heap.h"
#include "cricket-register.h"
#include "cricket-stack.h"

bool cricket_cr_function_name(uint64_t pc, const char **name)
{
    CUDBGResult res;
    const char *function_name;

    function_name = cuda_find_function_name_from_pc(pc, false);
    *name = function_name;
    return function_name != NULL;

cuda_error:
    fprintf(stderr, "Cuda Error: \"%s\"\n", cudbgGetErrorString(res));
    return false;
}

bool cricket_cr_sm_broken(CUDBGAPI cudbgAPI, uint32_t dev, uint32_t sm)
{
    uint64_t warp_mask;
    uint64_t warp_mask_broken;
    CUDBGResult res;
    res = cudbgAPI->readValidWarps(dev, sm, &warp_mask);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "%d:", __LINE__);
        goto cuda_error;
    }
    res = cudbgAPI->readBrokenWarps(dev, sm, &warp_mask_broken);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "%d:", __LINE__);
        goto cuda_error;
    }
    if (warp_mask != warp_mask_broken) {
        return false;
    }
    return true;
cuda_error:
    fprintf(stderr, "Cuda Error: \"%s\"\n", cudbgGetErrorString(res));
    return false;
}
bool cricket_cr_kernel_name(CUDBGAPI cudbgAPI, uint32_t dev, uint32_t sm,
                            uint32_t wp, const char **name)
{
    CUDBGResult res;
    uint64_t grid_id;
    CUDBGGridInfo info;
    const char *kernel_name;

    res = cudbgAPI->readGridId(dev, sm, wp, &grid_id);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "%d:", __LINE__);
        goto cuda_error;
    }

    res = cudbgAPI->getGridInfo(dev, grid_id, &info);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "%d:", __LINE__);
        goto cuda_error;
    }

    kernel_name = cuda_find_function_name_from_pc(info.functionEntry, false);
    *name = kernel_name;
    return kernel_name != NULL;

cuda_error:
    fprintf(stderr, "Cuda Error: \"%s\"\n", cudbgGetErrorString(res));
    return false;
}

static bool cricket_cr_gen_suffix(char **suffix, cricketWarpInfo *wi,
                                  uint32_t lane)
{
    size_t ret;
    if (lane == CRICKET_CR_NOLANE) {
        ret = asprintf(suffix, "-D%uS%uW%u", wi->dev, wi->sm, wi->warp);
    } else {
        ret =
            asprintf(suffix, "-D%uS%uW%uL%u", wi->dev, wi->sm, wi->warp, lane);
    }
    if (ret < 0) {
        fprintf(stderr, "cricket-cr: memory allocation failed\n");
        return false;
    }
    return true;
}

#define CRICKET_PROFILE 1
bool cricket_cr_rst_lane(CUDBGAPI cudbgAPI, cricketWarpInfo *wi, uint32_t lane,
                         const char *ckp_dir)
{
    CUDBGResult res;
    size_t register_size;
    void *reg_mem = NULL;
    void *stack_mem = NULL;
    char *suffix;
    bool ret = false;
    uint32_t stack_loc;
#ifdef CRICKET_PROFILE
    struct timeval b, c, d, e, g;
#endif

    if (!cricket_cr_gen_suffix(&suffix, wi, lane)) {
        return false;
    }

#ifdef CRICKET_PROFILE
    gettimeofday(&b, NULL);
#endif
    register_size = cricket_register_size(wi->dev_prop);
    if ((reg_mem = malloc(register_size)) == NULL) {
        fprintf(stderr,
                "cricket-cr (%u): error during memory allocation of size %lu\n",
                __LINE__, register_size);
        goto cleanup;
    }

    if (!cricket_file_read_mem(ckp_dir, CRICKET_DT_REGISTERS, suffix, reg_mem,
                               register_size)) {
        fprintf(stderr, "cricket-cr: error while setting registers\n");
        goto cleanup;
    }

#ifdef _CRICKET_DEBUG_CR_
    printf("register-data: ");
    for (int i = 0; i != register_size / sizeof(uint32_t); ++i) {
        printf("%08x ", ((uint32_t *)reg_mem)[i]);
    }
    printf("\n");
#endif

    stack_loc = ((uint32_t *)reg_mem)[cricket_stack_get_sp_regnum()];

#ifdef CRICKET_PROFILE
    gettimeofday(&c, NULL);
#endif

    if (!cricket_register_rst(cudbgAPI, wi->dev, wi->sm, wi->warp, lane,
                              reg_mem, wi->dev_prop)) {
        fprintf(stderr, "cricket-cr: error setting register data\n");
        goto cleanup;
    }
#ifdef CRICKET_PROFILE
    gettimeofday(&d, NULL);
#endif

    if ((stack_mem = malloc(wi->stack_size)) == NULL) {
        fprintf(stderr,
                "cricket-cr (%d): error during memory allocation of size %lu\n",
                __LINE__, register_size);
        goto cleanup;
    }

    if (!cricket_file_read_mem(ckp_dir, CRICKET_DT_STACK, suffix, stack_mem,
                               wi->stack_size)) {
        fprintf(stderr, "cricket-cr: error while setting stack memory\n");
        goto cleanup;
    }

#ifdef _CRICKET_DEBUG_CR_
    printf("stack-mem: ");
    for (int i = 0; i != wi->stack_size; ++i) {
        printf("%02x ", ((uint8_t *)stack_mem)[i]);
    }
    printf("\n");
#endif
#ifdef CRICKET_PROFILE
    gettimeofday(&e, NULL);
#endif

    if (!cricket_stack_set_mem(cudbgAPI, wi->dev, wi->sm, wi->warp, lane,
                               stack_mem, stack_loc, wi->stack_size)) {
        fprintf(stderr, "cricket-cr: error while retrieving stack memory\n");
        goto cleanup;
    }

#ifdef CRICKET_PROFILE
    gettimeofday(&g, NULL);
    double ct = ((double)((c.tv_sec * 1000000 + c.tv_usec) -
                          (b.tv_sec * 1000000 + b.tv_usec))) /
                1000000.;
    double dt = ((double)((d.tv_sec * 1000000 + d.tv_usec) -
                          (c.tv_sec * 1000000 + c.tv_usec))) /
                1000000.;
    double et = ((double)((e.tv_sec * 1000000 + e.tv_usec) -
                          (d.tv_sec * 1000000 + d.tv_usec))) /
                1000000.;
    double gt = ((double)((g.tv_sec * 1000000 + g.tv_usec) -
                          (e.tv_sec * 1000000 + e.tv_usec))) /
                1000000.;
    double comt = ((double)((g.tv_sec * 1000000 + g.tv_usec) -
                            (b.tv_sec * 1000000 + b.tv_usec))) /
                  1000000.;
    printf("lane time:\n\t\tPROFILE readreg: %f s\n\t\tPROFILE setreg: %f "
           "s\n\t\tPROFILE readstack: %f s\n\t\tPROFILE setstack: %f "
           "s\n\t\tPROFILE lanecomplete: %f s\n",
           ct, dt, et, gt, comt);
#endif
    ret = true;
cleanup:
    free(reg_mem);
    free(stack_mem);
    free(suffix);
    return ret;
}

#define CRICKET_INSTR_SSY_PREFIX 0xe2900000
bool cricket_cr_ckp_ssy(CUDBGAPI cudbgAPI, cricketWarpInfo *wi, uint32_t lane,
                        const char *ckp_dir)
{
    uint64_t pc;
    uint64_t virt_pc;
    uint64_t offset;
    uint64_t cur_instr;
    uint32_t rel_syn_pc;
    uint32_t syn_pc = 0;
    uint64_t sswarps;
    CUDBGResult res;

    res = cudbgAPI->readPC(wi->dev, wi->sm, wi->warp, lane, &pc);
    if (res != CUDBG_SUCCESS) {
        printf("%d: %s", __LINE__, cudbgGetErrorString(res));
        return false;
    }
    res = cudbgAPI->readVirtualPC(wi->dev, wi->sm, wi->warp, lane, &virt_pc);
    if (res != CUDBG_SUCCESS) {
        printf("%d: %s", __LINE__, cudbgGetErrorString(res));
        return false;
    }
    for (offset = 0L; offset <= pc; offset += 0x8) {
        res = cudbgAPI->readCodeMemory(wi->dev, virt_pc - offset, &cur_instr,
                                       sizeof(uint64_t));
        if (res != CUDBG_SUCCESS) {
            printf("%d: %s", __LINE__, cudbgGetErrorString(res));
            return false;
        }
        printf("instr: 0x%lx\n", cur_instr);
        if (((cur_instr >> 32) & 0xfff00000L) == CRICKET_INSTR_SSY_PREFIX) {
            rel_syn_pc = ((cur_instr >> 20) & 0x000ffffffffL);
            printf("rel_syn_pc: %x\n", rel_syn_pc);
            syn_pc = (pc - offset) + rel_syn_pc + 0x8;
            printf("syn_pc: %lx, is bigger: %d\n", syn_pc, syn_pc > pc);

            break;
        }
    }

    while (syn_pc > pc) {
        res = cudbgAPI->singleStepWarp(wi->dev, wi->sm, wi->warp, &sswarps);
        if (res != CUDBG_SUCCESS) {
            printf("%d: %s", __LINE__, cudbgGetErrorString(res));
        }
        res = cudbgAPI->readPC(wi->dev, wi->sm, wi->warp, lane, &pc);
        if (res != CUDBG_SUCCESS) {
            printf("%d: %s", __LINE__, cudbgGetErrorString(res));
        }
    }
    printf("new pc: %lx\n", pc);
    return true;
}

void cricket_cr_free_callstack(cricket_callstack *callstack)
{
    if (callstack == false)
        return;

    free(callstack->function_names);
    callstack->function_names = NULL;
    free(callstack->_packed_ptr);
    callstack->_packed_ptr = NULL;
}

#define _CRICKET_DEBUG_CR_ 1
bool cricket_cr_read_pc(cricketWarpInfo *wi, uint32_t lane, const char *ckp_dir,
                        cricket_callstack *callstack)
{
    CUDBGResult res;
    void *packed = NULL;
    size_t packed_size;
    uint32_t callstack_size;
    uint32_t valid_lanes;
    uint32_t active_lanes;
    char *suffix = NULL;
    bool ret = false;
    cricket_pc_data *pc_data;
    const char **function_names = NULL;
    size_t offset = 3 * sizeof(uint32_t);
    size_t i;

    if (callstack == NULL) {
        return false;
    }

    if (!cricket_cr_gen_suffix(&suffix, wi, lane)) {
        return false;
    }

    if (!cricket_file_read_mem_size(ckp_dir, CRICKET_DT_PC, suffix, &packed, 0,
                                    &packed_size)) {
        fprintf(stderr, "cricket-cr: error while reading pc memory\n");
        goto cleanup;
    }

    if (packed_size < offset) {
        fprintf(stderr, "cricket-cr: pc checkpoint file is corrupt: no "
                        "callstack_size\n");
        goto cleanup;
    }

    valid_lanes = *(uint32_t *)(packed);
    active_lanes = *(uint32_t *)(packed + sizeof(uint32_t));
    callstack_size = *(uint32_t *)(packed + 2 * sizeof(uint32_t));

#ifdef _CRICKET_DEBUG_CR_
    printf("valid_lanes: %x, active_lanes: %x, callstack_size: %x\n",
           valid_lanes, active_lanes, callstack_size);
#endif

    offset += callstack_size * sizeof(cricket_pc_data);
    if (packed_size < offset) {
        fprintf(stderr, "cricket-cr: pc checkpoint file is corrupt: too few "
                        "pc_data entries\n");
        goto cleanup;
    }

    pc_data = (cricket_pc_data *)(packed + 3 * sizeof(uint32_t));

    if (packed_size < offset + pc_data[callstack_size - 1].str_offset + 1) {
        fprintf(stderr, "cricket-cr: pc checkpoint file is corrupt: string "
                        "data to short\n");
        goto cleanup;
    }

    if ((function_names = malloc(callstack_size * sizeof(char *))) == NULL) {
        fprintf(stderr, "cricket-cr (%d): malloc failed\n", __LINE__);
        goto cleanup;
    }

    for (i = 0; i < callstack_size; ++i) {
        if (packed_size < offset + pc_data[i].str_offset) {
            fprintf(stderr, "cricket-cr: pc checkpoint file is corrupt: string "
                            "data to short\n");
            goto cleanup;
        }

        function_names[i] = packed + offset + pc_data[i].str_offset;
    }

#ifdef _CRICKET_DEBUG_CR_
    printf("callstack_size: %u\n", callstack_size);
    for (i = 0; i < callstack_size; ++i) {
        printf("relative: %lx, absolute: %lx, str_offset: %lx, function: %s\n",
               pc_data[i].relative, pc_data[i].absolute, pc_data[i].str_offset,
               function_names[i]);
    }
#endif

    callstack->valid_lanes = valid_lanes;
    callstack->active_lanes = active_lanes;
    callstack->callstack_size = callstack_size;
    callstack->pc = pc_data;
    callstack->function_names = function_names;
    free(suffix);
    return true;
cleanup:
    free(packed);
    free(function_names);
    free(suffix);
    return false;
}

// One specifc warp
bool cricket_cr_rst_subcall(CUDBGAPI cudbgAPI, cricketWarpInfo *wi,
                            cricket_callstack *callstack, int c_level,
                            cricket_jmptable_entry *cal, size_t cal_num,
                            uint64_t *cur_address)
{
    uint64_t jmptbl_address;
    uint32_t lanemask;
    bool ret = false;
    uint64_t sswarps;
    CUDBGResult res;

    if (!cricket_elf_get_jmptable_addr(
             cal, cal_num, callstack->pc[c_level].relative, &jmptbl_address)) {
        fprintf(stderr, "error getting jmptable adress\n");
        goto error;
    }

    res = cudbgAPI->readValidLanes(wi->dev, wi->sm, wi->warp, &lanemask);
    if (res != CUDBG_SUCCESS) {
        printf("%d:", __LINE__);
        goto error;
    }

    for (uint32_t lane = 0; lane != wi->dev_prop->numLanes; lane++) {
        if (lanemask & (1LU << lane)) {
            res = cudbgAPI->writeRegister(
                wi->dev, wi->sm, wi->warp, lane, CRICKET_JMX_ADDR_REG,
                ((uint32_t)(jmptbl_address - *cur_address - 0x8)));
            if (res != CUDBG_SUCCESS) {
                fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
                        cudbgGetErrorString(res));
                goto error;
            }
        }
    }

    res = cudbgAPI->singleStepWarp(wi->dev, wi->sm, wi->warp, &sswarps);
    if (res != CUDBG_SUCCESS) {
        printf("%d:", __LINE__);
        goto error;
    }

    res = cudbgAPI->singleStepWarp(wi->dev, wi->sm, wi->warp, &sswarps);
    if (res != CUDBG_SUCCESS) {
        printf("%d:", __LINE__);
        goto error;
    }
    *cur_address = jmptbl_address + 0x8;

    ret = true;
error:
    return ret;
}
// One specifc warp
bool cricket_cr_rst_ssy(CUDBGAPI cudbgAPI, cricketWarpInfo *wi,
                        cricket_callstack *callstack, int c_level,
                        cricket_jmptable_entry *ssy, size_t ssy_num,
                        uint64_t *cur_address)
{
    uint64_t relative_ssy;
    uint64_t jmptbl_address;
    uint32_t lanemask;
    bool ret = false;
    uint64_t sswarps;
    uint64_t cmp_address;
    CUDBGResult res;

    if (!cricket_elf_pc_info(callstack->function_names[c_level],
                             callstack->pc[c_level].relative, &relative_ssy,
                             NULL)) {
        fprintf(stderr, "cricket-restore: getting ssy point failed\n");
        goto error;
    }
    cmp_address = callstack->pc[c_level].relative;
    if (relative_ssy % (4 * 8) == 0) {
        cmp_address -= 0x8;
    }

    if (relative_ssy >= cmp_address) {
        if (!cricket_elf_get_jmptable_addr(ssy, ssy_num, relative_ssy,
                                           &jmptbl_address)) {
            fprintf(stderr, "error getting jmptable adress\n");
            goto error;
        }

        res = cudbgAPI->readValidLanes(wi->dev, wi->sm, wi->warp, &lanemask);
        if (res != CUDBG_SUCCESS) {
            printf("%d:", __LINE__);
            goto error;
        }

        for (uint32_t lane = 0; lane != wi->dev_prop->numLanes; lane++) {
            if (lanemask & (1LU << lane)) {
                res = cudbgAPI->writeRegister(
                    wi->dev, wi->sm, wi->warp, lane, CRICKET_JMX_ADDR_REG,
                    ((uint32_t)(jmptbl_address - *cur_address - 0x8)));
                if (res != CUDBG_SUCCESS) {
                    fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
                            cudbgGetErrorString(res));
                    goto error;
                }
            }
        }

        res = cudbgAPI->singleStepWarp(wi->dev, wi->sm, wi->warp, &sswarps);
        if (res != CUDBG_SUCCESS) {
            printf("%d:", __LINE__);
            goto error;
        }

        res = cudbgAPI->singleStepWarp(wi->dev, wi->sm, wi->warp, &sswarps);
        if (res != CUDBG_SUCCESS) {
            printf("%d:", __LINE__);
            goto error;
        }
        *cur_address = jmptbl_address + 0x8;
        if (*cur_address % (4 * 8) == 0) {
            *cur_address += 0x8;
        }
        printf("restored ssy\n");
    }
    ret = true;
error:
    return ret;
}

bool cricket_cr_rst_pc(CUDBGAPI cudbgAPI, cricketWarpInfo *wi,
                       uint32_t lane_param, cricket_callstack *callstack)
{
    CUDBGResult res;
    bool ret = false;
    uint32_t lane;
    uint64_t address;
    uint32_t predicate;

    if (lane_param == CRICKET_CR_NOLANE) {
        lane = 0;
    } else {
        lane = lane_param;
    }
    if (!(callstack->valid_lanes & (1 << lane))) {
        fprintf(stderr, "lane %d is not valid\n", lane);
    }

    if (callstack->callstack_size == 1) {
        address = callstack->pc[0].relative - 0x20;
    } else if (callstack->callstack_size == 2) {
        if (callstack->active_lanes != callstack->valid_lanes) {
            fprintf(stderr, "cricket-cr: divergent threads in call levels > 1 "
                            "are not allowed!\n");
            return false;
        }
        address = callstack->pc[0].absolute;

        /*if (callstack->active_lanes & (1<<lane)) {
            address = callstack->pc[0].absolute;
        } else {
            address =
        callstack->pc[0].absolute-callstack->pc[0].relative+0x1150;
        }*/
    } else {
        fprintf(stderr, "cricket-cr: callstacks greater than 2 cannot be "
                        "restored\n");
        return false;
    }

    if (address > (1ULL << 32)) {
        fprintf(stderr, "cricket-cr: pc value is too large to be restored!\n");
        goto cleanup;
    }

    if (callstack->active_lanes & (1 << lane)) {
        predicate = 0;
    } else {
        predicate = 1;
    }

    res = cudbgAPI->writePredicates(wi->dev, wi->sm, wi->warp, lane, 1,
                                    &predicate);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
                cudbgGetErrorString(res));
        return false;
    }

    do {
        res =
            cudbgAPI->writeRegister(wi->dev, wi->sm, wi->warp, lane,
                                    CRICKET_JMX_ADDR_REG, ((uint32_t)address));

        if (res != CUDBG_SUCCESS) {
            fprintf(stderr, "cricket-cr (%d) D%uS%uW%uL%u: %s, retrying...\n",
                    __LINE__, wi->dev, wi->sm, wi->warp, lane,
                    cudbgGetErrorString(res));
        }
    } while (res != CUDBG_SUCCESS);

    /*res = cudbgAPI->writeRegister(wi->dev, wi->sm, wi->warp, lane,
    CRICKET_JMX_ADDR_REG+1, (uint32_t)0x0);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
    cudbgGetErrorString(res));
        goto cleanup;
    }*/

    ret = true;
cleanup:
    return ret;
}

bool cricket_cr_callstack(CUDBGAPI cudbgAPI, cricketWarpInfo *wi,
                          uint32_t lane_param, cricket_callstack *callstack)
{
    uint32_t base_addr;
    uint64_t call_instr;
    cricket_pc_data *pc_data = NULL;
    const char **function_names = NULL;
    uint32_t i;
    uint32_t callstack_size;
    uint32_t active_lanes;
    uint32_t valid_lanes;
    size_t str_offset = 0;
    bool ret = false;
    uint32_t lane;
    size_t offset = 0;
    CUDBGResult res;

    if (callstack == NULL)
        return false;

    res = cudbgAPI->readValidLanes(wi->dev, wi->sm, wi->warp, &valid_lanes);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
                cudbgGetErrorString(res));
        return false;
    }

    res = cudbgAPI->readActiveLanes(wi->dev, wi->sm, wi->warp, &active_lanes);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
                cudbgGetErrorString(res));
        return false;
    }

    if (lane_param == CRICKET_CR_NOLANE) {
        lane = 0;
        for (uint32_t lane = 0; lane != wi->dev_prop->numLanes; lane++) {
            if (active_lanes & (1LU << lane)) {
                lane = i;
                break;
            }
        }
    } else {
        lane = lane_param;
    }

    res = cudbgAPI->readCallDepth(wi->dev, wi->sm, wi->warp, lane,
                                  &callstack_size);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
                cudbgGetErrorString(res));
        return false;
    }
    callstack_size++;

    if ((function_names = malloc(callstack_size * sizeof(char *))) == NULL) {
        fprintf(stderr, "cricket-cr (%d): malloc failed\n", __LINE__);
        return false;
    }
    if ((pc_data = malloc(callstack_size * sizeof(cricket_pc_data))) == NULL) {
        fprintf(stderr, "cricket-cr (%d): malloc failed\n", __LINE__);
        goto cleanup;
    }

    res =
        cudbgAPI->readPC(wi->dev, wi->sm, wi->warp, lane, &pc_data[0].relative);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
                cudbgGetErrorString(res));
        goto cleanup;
    }

    res = cudbgAPI->readVirtualPC(wi->dev, wi->sm, wi->warp, lane,
                                  &pc_data[0].virt);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
                cudbgGetErrorString(res));
        goto cleanup;
    }

    if (!cricket_cr_function_name(pc_data[0].virt, &function_names[0])) {
        fprintf(stderr, "cricket-cr: error getting function name\n");
        goto cleanup;
    }
    printf("relative %lx, virtual %lx\n", pc_data[0].relative, pc_data[0].virt);

    pc_data[0].str_offset = 0;
    str_offset = strlen(function_names[0]) + 1;

    for (i = 1; i < callstack_size; ++i) {
        res = cudbgAPI->readReturnAddress(wi->dev, wi->sm, wi->warp, lane,
                                          i - 1, &pc_data[i].relative);
        if (res != CUDBG_SUCCESS) {
            fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
                    cudbgGetErrorString(res));
            goto cleanup;
        }

        res = cudbgAPI->readVirtualReturnAddress(wi->dev, wi->sm, wi->warp,
                                                 lane, i - 1, &pc_data[i].virt);
        if (res != CUDBG_SUCCESS) {
            fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
                    cudbgGetErrorString(res));
            goto cleanup;
        }

        if (!cricket_cr_function_name(pc_data[i].virt, &function_names[i])) {
            fprintf(stderr, "cricket-cr: error getting function name\n");
            goto cleanup;
        }
        pc_data[i].str_offset = str_offset;
        str_offset += strlen(function_names[i]) + 1;

        res = cudbgAPI->readCodeMemory(wi->dev,
                                       pc_data[i].virt - sizeof(uint64_t),
                                       &call_instr, sizeof(uint64_t));
        if (res != CUDBG_SUCCESS) {
            fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
                    cudbgGetErrorString(res));
            goto cleanup;
        }

        base_addr = (call_instr >> 20);

        pc_data[i - 1].absolute = base_addr + pc_data[i - 1].relative;
    }
    pc_data[callstack_size - 1].absolute = 0;

    callstack->active_lanes = active_lanes;
    callstack->valid_lanes = valid_lanes;
    callstack->callstack_size = callstack_size;
    callstack->pc = pc_data;
    callstack->function_names = function_names;
    callstack->_packed_ptr = NULL;

    ret = true;
    return ret;

cleanup:
    free(function_names);
    free(pc_data);
    return ret;
}

bool cricket_cr_ckp_pc(CUDBGAPI cudbgAPI, cricketWarpInfo *wi,
                       uint32_t lane_param, const char *ckp_dir,
                       cricket_callstack *callstack)
{
    char *suffix;
    void *packed = NULL;
    size_t packed_size;
    bool ret = false;
    size_t str_size;
    size_t offset = 0;
    uint32_t lane;
    size_t i;
    CUDBGResult res;
    if (callstack == NULL || wi == NULL || ckp_dir == NULL)
        return false;

    if (lane_param == CRICKET_CR_NOLANE) {
        lane = 0;
    } else {
        lane = lane_param;
    }
    str_size =
        callstack->pc[callstack->callstack_size - 1].str_offset +
        strlen(callstack->function_names[callstack->callstack_size - 1]) + 1;

    packed_size = 3 * sizeof(uint32_t) +
                  callstack->callstack_size * sizeof(cricket_pc_data) +
                  str_size;
    if ((packed = malloc(packed_size)) == NULL) {
        goto cleanup;
    }

    memcpy(packed, &callstack->valid_lanes, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    memcpy(packed + offset, &callstack->active_lanes, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    memcpy(packed + offset, &callstack->callstack_size, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    memcpy(packed + offset, callstack->pc,
           callstack->callstack_size * sizeof(cricket_pc_data));
    offset += callstack->callstack_size * sizeof(cricket_pc_data);
    for (i = 0; i < callstack->callstack_size; ++i) {
        strcpy(packed + offset + callstack->pc[i].str_offset,
               callstack->function_names[i]);
    }

#ifdef _CRICKET_DEBUG_CR_
    printf("callstack_size: %u\n", callstack->callstack_size);
    for (i = 0; i < callstack->callstack_size; ++i) {
        printf("relative: %lx, absolute: %lx, str_offset: %lx, function: %s\n",
               callstack->pc[i].relative, callstack->pc[i].absolute,
               callstack->pc[i].str_offset, callstack->function_names[i]);
    }
#endif

    if (!cricket_cr_gen_suffix(&suffix, wi, lane_param)) {
        goto cleanup;
    }

    if (!cricket_file_store_mem(ckp_dir, CRICKET_DT_PC, suffix, packed,
                                packed_size)) {
        fprintf(stderr, "cricket-cr: error writing pc\n");
        goto cleanup;
    }

    ret = true;

cleanup:
    free(packed);
    free(suffix);
    return ret;
}
/*
bool cricket_cr_ckp_pc(CUDBGAPI cudbgAPI, cricketWarpInfo *wi, uint32_t
lane_param, const char *ckp_dir)
{
    uint32_t base_addr;
    uint64_t call_instr;
    cricket_pc_data *pc_data = NULL;
    const char **function_names = NULL;
    uint32_t i;
    uint32_t callstack_size;
    uint32_t active_lanes;
    uint32_t valid_lanes;
    uint64_t virt_pc;
    size_t str_offset = 0;
    char *suffix;
    void *packed = NULL;
    size_t packed_size;
    bool ret = false;
    uint32_t lane;
    size_t offset = 0;
    CUDBGResult res;

    if (lane_param == CRICKET_CR_NOLANE) {
        lane = 0;
    } else {
        lane = lane_param;
    }

    res = cudbgAPI->readValidLanes(wi->dev, wi->sm, wi->warp, &valid_lanes);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
cudbgGetErrorString(res));
        return false;
    }

    res = cudbgAPI->readActiveLanes(wi->dev, wi->sm, wi->warp, &active_lanes);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
cudbgGetErrorString(res));
        return false;
    }

    res = cudbgAPI->readCallDepth(wi->dev, wi->sm, wi->warp, lane
,&callstack_size);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
cudbgGetErrorString(res));
        return false;
    }
    callstack_size++;

    if ((function_names = malloc(callstack_size*sizeof(char*))) == NULL) {
        fprintf(stderr, "cricket-cr (%d): malloc failed\n", __LINE__);
        return false;
    }
    if ((pc_data = malloc(callstack_size*sizeof(cricket_pc_data))) == NULL) {
        fprintf(stderr, "cricket-cr (%d): malloc failed\n", __LINE__);
        goto cleanup;
    }

    res = cudbgAPI->readPC(wi->dev, wi->sm, wi->warp, lane,
&pc_data[0].relative);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
cudbgGetErrorString(res));
        goto cleanup;
    }

    res = cudbgAPI->readVirtualPC(wi->dev, wi->sm, wi->warp, lane, &virt_pc);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
cudbgGetErrorString(res));
        goto cleanup;
    }

    if (!cricket_cr_function_name(virt_pc, &function_names[0])) {
        fprintf(stderr, "cricket-cr: error getting function name\n");
        goto cleanup;
    }

    pc_data[0].str_offset = 0;
    str_offset = strlen(function_names[0])+1;


    for (i=1; i < callstack_size; ++i) {

        res = cudbgAPI->readReturnAddress(wi->dev, wi->sm, wi->warp, lane, i-1,
&pc_data[i].relative);
        if (res != CUDBG_SUCCESS) {
            fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
cudbgGetErrorString(res));
            goto cleanup;
        }

        res = cudbgAPI->readVirtualReturnAddress(wi->dev, wi->sm, wi->warp,
lane, i-1, &virt_pc);
        if (res != CUDBG_SUCCESS) {
            fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
cudbgGetErrorString(res));
            goto cleanup;
        }

        if (!cricket_cr_function_name(virt_pc, &function_names[i])) {
            fprintf(stderr, "cricket-cr: error getting function name\n");
            goto cleanup;
        }
        pc_data[i].str_offset = str_offset;
        str_offset += strlen(function_names[i])+1;

        res = cudbgAPI->readCodeMemory(wi->dev, virt_pc-sizeof(uint64_t),
&call_instr, sizeof(uint64_t));
        if (res != CUDBG_SUCCESS) {
            fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
cudbgGetErrorString(res));
            goto cleanup;
        }

        base_addr = (call_instr >> 20);

        pc_data[i-1].absolute = base_addr + pc_data[i-1].relative;

    }
    pc_data[callstack_size-1].absolute = 0;


    packed_size =
3*sizeof(uint32_t)+callstack_size*sizeof(cricket_pc_data)+str_offset;
    if ((packed = malloc(packed_size)) == NULL) {
        goto cleanup;
    }

    memcpy(packed, &valid_lanes, sizeof(uint32_t));
    offset+=sizeof(uint32_t);
    memcpy(packed+offset, &active_lanes, sizeof(uint32_t));
    offset+=sizeof(uint32_t);
    memcpy(packed+offset, &callstack_size, sizeof(uint32_t));
    offset+=sizeof(uint32_t);
    memcpy(packed+offset, pc_data, callstack_size*sizeof(cricket_pc_data));
    offset+=callstack_size*sizeof(cricket_pc_data);
    for (i = 0; i < callstack_size; ++i) {
        strcpy(packed+offset+pc_data[i].str_offset, function_names[i]);
    }


#ifdef _CRICKET_DEBUG_CR_
    printf("callstack_size: %u\n", callstack_size);
    for (i=0; i < callstack_size; ++i) {
        printf("relative: %lx, absolute: %lx, str_offset: %lx, function: %s\n",
pc_data[i].relative,
                                                                                pc_data[i].absolute,
                                                                                pc_data[i].str_offset,
                                                                                function_names[i]);
    }
#endif


    if (!cricket_cr_gen_suffix(&suffix, wi, lane_param)) {
        goto cleanup;
    }

    if (!cricket_file_store_mem(ckp_dir, CRICKET_DT_PC, suffix, packed,
packed_size)) {
        fprintf(stderr, "cricket-cr: error writing pc\n");
        free(suffix);
        goto cleanup;
    }

    ret = true;

 cleanup:
    free(function_names);
    free(pc_data);
    free(packed);
    free(suffix);
    return ret;

}
*/

bool cricket_cr_make_checkpointable(CUDBGAPI cudbgAPI, cricketWarpInfo *wi,
                                    cricket_function_info *fi, size_t fi_num,
                                    cricket_callstack *callstack)
{
    CUDBGResult res;
    uint64_t rel_ssy, rel_pbk;
    cricket_function_info *the_fi = NULL;
    bool ret = false;
    size_t i;
    bool joined = false;
    bool stepup = false;
    size_t callstack_bku = callstack->callstack_size;
    for (i = callstack->callstack_size - 1; i + 1 > 0; --i) {
        if (!cricket_elf_get_fun_info(fi, fi_num, callstack->function_names[i],
                                      &the_fi)) {
            fprintf(stderr, "cricket-cr: failed to get fun_info\n");
            goto cleanup;
        }
        if (the_fi == NULL) {
            fprintf(stderr, "cricket-cr: no info for function %s available\n",
                    callstack->function_names[i]);
            goto cleanup;
        }
        printf("function \"%s\" has %sroom (%lu slots)\n",
               callstack->function_names[i], (the_fi->has_room ? "" : "no "),
               the_fi->room);
        if (i == callstack->callstack_size - 1 && the_fi->room == 0) {
            fprintf(stderr, "cricket-cr: There is no room in the top level "
                            "function (i.e. the kernel). This kernel can thus "
                            "never be restored!\n");
            goto cleanup;
        }
        if (!the_fi->has_room) {
            printf("function %s does not have enough room. Subcalls and "
                   "synchronization points in this function cannot be "
                   "restored\n",
                   callstack->function_names[i]);
            if (i > 1) {
                printf("no room in %s. going up to %lx (+%lx)...\n",
                       callstack->function_names[i], callstack->pc[i].virt,
                       callstack->pc[i].relative);
                res = cudbgAPI->resumeWarpsUntilPC(wi->dev, wi->sm,
                                                   (0x1 << wi->warp),
                                                   callstack->pc[i].virt + 0x8);
                if (res != CUDBG_SUCCESS) {
                    printf("cuda-cr (%d): resumeWarpsUntilPC: ", __LINE__);
                    printf("Cuda Error: \"%s\"\n", cudbgGetErrorString(res));
                    goto cleanup;
                }

                if (!cricket_cr_sm_broken(cudbgAPI, wi->dev, wi->sm)) {
                    printf("waiting for sm to break...\n");
                    while (!cricket_cr_sm_broken(cudbgAPI, wi->dev, wi->sm)) {
                        usleep(500);
                    }
                }
                stepup = true;
            }

            if (!cricket_elf_pc_info(callstack->function_names[i],
                                     callstack->pc[i].relative, &rel_ssy,
                                     &rel_pbk)) {
                fprintf(stderr, "cricket: pc info failed\n");
                goto cleanup;
            }
            printf("ssy: %lx > relative: %lx ?\n", rel_ssy,
                   callstack->pc[i].relative);
            if (rel_ssy > callstack->pc[i].relative) {
                printf("joining to pc %lx\n", callstack->pc[i].virt -
                                                  callstack->pc[i].relative +
                                                  rel_ssy);

                res = cudbgAPI->resumeWarpsUntilPC(
                    wi->dev, wi->sm, (0x1 << wi->warp),
                    callstack->pc[i].virt - callstack->pc[i].relative +
                        rel_ssy);
                if (res != CUDBG_SUCCESS) {
                    printf("cuda-cr (%d): resumeWarpsUntilPC: ", __LINE__);
                    printf("Cuda Error: \"%s\"\n", cudbgGetErrorString(res));
                    goto cleanup;
                }
                if (!cricket_cr_sm_broken(cudbgAPI, wi->dev, wi->sm)) {
                    printf("waiting for sm to break...\n");
                    while (!cricket_cr_sm_broken(cudbgAPI, wi->dev, wi->sm)) {
                        usleep(500);
                    }
                }
                printf("joined divergent threads\n");
                joined = true;
            }
            if (rel_pbk > callstack->pc[i].relative) {
                printf("breaking to pc %lx\n", callstack->pc[i].virt -
                                                   callstack->pc[i].relative +
                                                   rel_pbk);

                res = cudbgAPI->resumeWarpsUntilPC(
                    wi->dev, wi->sm, (0x1 << wi->warp),
                    callstack->pc[i].virt - callstack->pc[i].relative +
                        rel_pbk);
                if (res != CUDBG_SUCCESS) {
                    printf("cuda-cr (%d): resumeWarpsUntilPC: ", __LINE__);
                    printf("Cuda Error: \"%s\"\n", cudbgGetErrorString(res));
                    goto cleanup;
                }
                if (!cricket_cr_sm_broken(cudbgAPI, wi->dev, wi->sm)) {
                    printf("waiting for sm to break...\n");
                    while (!cricket_cr_sm_broken(cudbgAPI, wi->dev, wi->sm)) {
                        usleep(500);
                    }
                }
                printf("break'ed divergent threads\n");
                joined = true;
            }

            if (joined || stepup) {
                cricket_cr_free_callstack(callstack);

                if (!cricket_cr_callstack(cudbgAPI, wi, CRICKET_CR_NOLANE,
                                          callstack)) {
                    fprintf(stderr, "cricket-cr: failed to get callstack\n");
                    goto cleanup;
                }
                printf("new callstack size %d\n", callstack->callstack_size);
                if (callstack->callstack_size != callstack_bku - i) {
                    fprintf(stderr, "cricket-cr: new callstack has wrong "
                                    "size\n");
                    goto cleanup;
                }
                if (joined &&
                    callstack->valid_lanes != callstack->active_lanes) {
                    fprintf(stderr, "cricket-cr: joning failed, threads still "
                                    "divergent @ rel PC %lx\n",
                            callstack->pc[0].relative);
                    goto cleanup;
                }
            }
            break;
        }
    }

    if (stepup) {
        printf("warp was stepped up\n");
    } else {
        printf("no up stepping required\n");
    }
    if (joined) {
        printf("threads were joined\n");
    } else {
        printf("no joining required\n");
    }
    printf("threads are now checkpointable\n");

    ret = true;
cleanup:
    return ret;
}

/*
bool cricket_cr_join_threads(CUDBGAPI cudbgAPI, uint32_t dev, uint32_t sm,
uint32_t wp)
{
    CUDBGResult res;
    uint64_t rel_pc;
    uint64_t virt_pc;
    uint64_t rel_ssy;
    const char *fn;
    uint32_t active_lanemask, valid_lanemask;
    bool ret = false;

    res = cudbgAPI->readVirtualPC(dev,sm,wp,0, &virt_pc);
    if (res != CUDBG_SUCCESS) {
        printf("%d:", __LINE__);
        printf("Cuda Error: \"%s\"\n", cudbgGetErrorString(res));
        goto cleanup;
    }
    res = cudbgAPI->readPC(dev,sm,wp,0, &rel_pc);
    if (res != CUDBG_SUCCESS) {
        printf("%d:", __LINE__);
        printf("Cuda Error: \"%s\"\n", cudbgGetErrorString(res));
        goto cleanup;
    }

    if (!cricket_cr_function_name(virt_pc, &fn)) {
        fprintf(stderr, "cricket-cr: error getting function name\n");
        goto cleanup;
    }
    if (!cricket_elf_pc_info(fn, rel_pc, &rel_ssy, NULL)) {
        fprintf(stderr, "cricket: pc info failed\n");
        goto cleanup;
    }
    printf("ssy: %lx, rel_pc: %lx, virt_pc: %lx\n", rel_ssy, rel_pc, virt_pc);
    if (rel_ssy == 0 || rel_ssy < rel_pc) {
        fprintf(stderr, "cricket-cr: no applicable ssy instruction found.
Threads may be diverged due to something else (propably unsupported)\n");
        goto cleanup;
    }
    res = cudbgAPI->resumeWarpsUntilPC (dev, sm, (0x1 << wp),
virt_pc-rel_pc+rel_ssy );
    if (res != CUDBG_SUCCESS) {
        printf("cuda-cr (%d): resumeWarpsUntilPC: ", __LINE__);
        printf("Cuda Error: \"%s\"\n", cudbgGetErrorString(res));
        goto cleanup;
    }

    res = cudbgAPI->readActiveLanes(dev, sm, wp, &active_lanemask);
    if (res != CUDBG_SUCCESS) {
        printf("cuda-cr (%d): resumeWarpsUntilPC: ", __LINE__);
        printf("Cuda Error: \"%s\"\n", cudbgGetErrorString(res));
        goto cleanup;
    }
    res = cudbgAPI->readValidLanes(dev, sm, wp, &valid_lanemask);
    if (res != CUDBG_SUCCESS) {
        printf("cuda-cr (%d): resumeWarpsUntilPC: ", __LINE__);
        printf("Cuda Error: \"%s\"\n", cudbgGetErrorString(res));
        goto cleanup;
    }

    if (valid_lanemask != active_lanemask) {
        fprintf(stderr, "cricket-cr: joining threads failed :( (active: %x,
valid: %x)\n", active_lanemask, valid_lanemask);
        goto cleanup;
    }

    ret = true;
 cleanup:
    return ret;
}*/

#undef _CRICKET_DEBUG_CR_
/*#define _CRICKET_DEBUG_CR_ 1
bool cricket_cr_ckp_pc(CUDBGAPI cudbgAPI, cricketWarpInfo *wi, uint32_t lane,
const char *ckp_dir)
{
    uint64_t virt_ret_addr;
    uint32_t base_addr;
    uint64_t call_instr;
    struct cricket_pc_data pc_data = {0};
    char *suffix;
    CUDBGResult res;

    res = cudbgAPI->readCallDepth(wi->dev, wi->sm, wi->warp, lane
,&pc_data.call_depth);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
cudbgGetErrorString(res));
        return false;
    }

    if (pc_data.call_depth != 1) {
        fprintf(stderr, "cricket-cr: call-depth unequal 1 currently not
supported\n");
        return false;
    }

    res = cudbgAPI->readVirtualReturnAddress(wi->dev, wi->sm, wi->warp, lane,
pc_data.call_depth-1, &virt_ret_addr);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
cudbgGetErrorString(res));
        return false;
    }

    res = cudbgAPI->readReturnAddress(wi->dev, wi->sm, wi->warp, lane,
pc_data.call_depth-1, &pc_data.relative_ret_addr);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
cudbgGetErrorString(res));
        return false;
    }

    res = cudbgAPI->readCodeMemory(wi->dev, virt_ret_addr-sizeof(uint64_t),
&call_instr, sizeof(uint64_t));
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
cudbgGetErrorString(res));
        return false;
    }

    base_addr = (call_instr >> 20);

    res = cudbgAPI->readPC(wi->dev, wi->sm, wi->warp, lane,
&pc_data.relative_addr);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
cudbgGetErrorString(res));
        return false;
    }

    pc_data.absolute_addr = base_addr + pc_data.relative_addr;

#ifdef _CRICKET_DEBUG_CR_
    printf("call depth: %x\n", pc_data.call_depth);
    printf("virtual return address:%lx\n", virt_ret_addr);
    printf("relative return address:%lx\n", pc_data.relative_ret_addr);
    printf("call instruction: %lx\n", call_instr);
    printf("base_addr: %lx\n", base_addr);
    printf("relative addr: %lx\n", pc_data.relative_addr);
    printf("absolute pc: %lx\n", pc_data.absolute_addr);
#endif


    if (!cricket_cr_gen_suffix(&suffix, wi, lane)) {
        return false;
    }

    if (!cricket_file_store_mem(ckp_dir, CRICKET_DT_PC, suffix, &pc_data,
sizeof(struct cricket_pc_data))) {
        fprintf(stderr, "cricket-cr: error writing pc\n");
        free(suffix);
        return false;
    }

    free(suffix);
    return true;

}
#undef _CRICKET_DEBUG_CR_*/

/* stores stack, registers and PC */
#define CRICKET_PROFILE 1
bool cricket_cr_ckp_lane(CUDBGAPI cudbgAPI, cricketWarpInfo *wi, uint32_t lane,
                         const char *ckp_dir)
{
    CUDBGResult res;
    size_t register_size;
    void *mem = NULL;
    char *suffix;
    bool ret = false;
#ifdef CRICKET_PROFILE
    struct timeval b, c, d, e, g;
#endif

    if (!cricket_cr_gen_suffix(&suffix, wi, lane)) {
        return false;
    }

    if ((mem = malloc(wi->stack_size)) == NULL) {
        fprintf(stderr,
                "cricket-cr (%d): error during memory allocation of size %d\n",
                __LINE__, wi->stack_size);
        goto cleanup;
    }
#ifdef CRICKET_PROFILE
    gettimeofday(&b, NULL);
#endif

    if (!cricket_stack_get_mem(cudbgAPI, wi->dev, wi->sm, wi->warp, lane, mem,
                               wi->stack_size)) {
        fprintf(stderr, "cricket-cr: error while retrieving stack memory\n");
        goto cleanup;
    }

#ifdef _CRICKET_DEBUG_CR_
    printf("stack-mem: ");
    for (int i = 0; i != wi->stack_size; ++i) {
        printf("%02x ", ((uint8_t *)mem)[i]);
    }
    printf("\n");
#endif
#ifdef CRICKET_PROFILE
    gettimeofday(&c, NULL);
#endif

    if (!cricket_file_store_mem(ckp_dir, CRICKET_DT_STACK, suffix, mem,
                                wi->stack_size)) {
        fprintf(stderr, "cricket-cr: error writing stack memory\n");
        goto cleanup;
    }
    register_size = cricket_register_size(wi->dev_prop);
    if ((mem = realloc(mem, register_size)) == NULL) {
        fprintf(stderr, "cricket-cr (%lu): error during memory allocation of "
                        "size %lu\n",
                __LINE__, register_size);
        goto cleanup;
    }
#ifdef CRICKET_PROFILE
    gettimeofday(&d, NULL);
#endif

    if (!cricket_register_ckp(cudbgAPI, wi->dev, wi->sm, wi->warp, lane, mem,
                              wi->dev_prop)) {
        fprintf(stderr, "cricket-cr: error retrieving register data\n");
        goto cleanup;
    }

#ifdef _CRICKET_DEBUG_CR_
    printf("register-data: ");
    for (int i = 0; i != register_size / sizeof(uint32_t); ++i) {
        printf("%08x ", ((uint32_t *)mem)[i]);
    }
    printf("\n");
#endif
#ifdef CRICKET_PROFILE
    gettimeofday(&e, NULL);
#endif

    if (!cricket_file_store_mem(ckp_dir, CRICKET_DT_REGISTERS, suffix, mem,
                                register_size)) {
        fprintf(stderr, "cricket-cr: error writing registers\n");
        goto cleanup;
    }

    if ((mem = realloc(mem, sizeof(uint64_t))) == NULL) {
        fprintf(stderr,
                "cricket-cr (%u): error during memory allocation of size %lu\n",
                __LINE__, sizeof(uint64_t));
        goto cleanup;
    }
#ifdef CRICKET_PROFILE
    gettimeofday(&g, NULL);
    double ct = ((double)((c.tv_sec * 1000000 + c.tv_usec) -
                          (b.tv_sec * 1000000 + b.tv_usec))) /
                1000000.;
    double dt = ((double)((d.tv_sec * 1000000 + d.tv_usec) -
                          (c.tv_sec * 1000000 + c.tv_usec))) /
                1000000.;
    double et = ((double)((e.tv_sec * 1000000 + e.tv_usec) -
                          (d.tv_sec * 1000000 + d.tv_usec))) /
                1000000.;
    double gt = ((double)((g.tv_sec * 1000000 + g.tv_usec) -
                          (e.tv_sec * 1000000 + e.tv_usec))) /
                1000000.;
    double comt = ((double)((g.tv_sec * 1000000 + g.tv_usec) -
                            (b.tv_sec * 1000000 + b.tv_usec))) /
                  1000000.;
    printf("lane time:\n\t\tPROFILE getstack: %f s\n\t\tPROFILE storestack: %f "
           "s\n\t\tPROFILE getreg: %f s\n\t\tPROFILE storereg: %f "
           "s\n\t\tPROFILE lanecomplete: %f s\n",
           ct, dt, et, gt, comt);
#endif

    ret = true;
cleanup:
    free(suffix);
    free(mem);
    return ret;
}

bool cricket_cr_rst_params(CUDBGAPI cudbgAPI, const char *ckp_dir,
                           cricket_elf_info *elf_info, uint32_t dev,
                           uint32_t sm, uint32_t warp)
{
    CUDBGResult res;
    uint8_t *param_mem = NULL;
    void *param_data = NULL;
    size_t heapsize;
    char heap_suffix[8];
    bool ret = false;
    /* Parameters are the same for all warps so just use warp 0
     * TODO: use first valid warp, because warp 0 may not be in use (is that
     * possible?)
     */
    if ((param_mem = malloc(elf_info->param_size)) == NULL)
        return false;

    if (!cricket_file_read_mem(ckp_dir, CRICKET_DT_PARAM, NULL, param_mem,
                               elf_info->param_size)) {
        printf("error reading param\n");
        goto cleanup;
    }

#ifdef _CRICKET_DEBUG_CR_
    printf("param-mem: ");
    for (int i = 0; i != elf_info->param_size; ++i) {
        printf("%02x ", param_mem[i]);
    }
    printf("\n");
#endif

    res = cudbgAPI->writeParamMemory(dev, sm, warp,
                                     (uint64_t)elf_info->param_addr, param_mem,
                                     (uint32_t)elf_info->param_size);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d):%s", __LINE__,
                cudbgGetErrorString(res));
        goto cleanup;
    }
    for (int i = 0; i != elf_info->param_num; ++i) {
        if (elf_info->params[i].size != 8)
            continue;

        sprintf(heap_suffix, "-P%u", elf_info->params[i].index);

        if (!cricket_file_exists(ckp_dir, CRICKET_DT_HEAP, heap_suffix)) {
            printf("no checkpoint file for parameter %u\n", i);
            continue;
        }
        param_data = NULL;
        if (!cricket_file_read_mem_size(ckp_dir, CRICKET_DT_HEAP, heap_suffix,
                                        &param_data, 0, &heapsize)) {
            printf("cricket error while reading heap param data\n");
            goto cleanup;
        }

#ifdef _CRICKET_DEBUG_CR_
        printf("heap param %u: %llx (%u)\n", i,
               *(void **)(param_mem + elf_info->params[i].offset), heapsize);
        printf("param-data for param %u: ", i);
        for (int i = 0; i != heapsize; ++i) {
            printf("%02x ", ((uint8_t *)param_data)[i]);
        }
        printf("\n");
#endif

        res = cudbgAPI->writeGlobalMemory(
            *(uint64_t *)(param_mem + elf_info->params[i].offset), param_data,
            heapsize);
        if (res != CUDBG_SUCCESS) {
            fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
                    cudbgGetErrorString(res));
            goto cleanup;
        }
        free(param_data);
        param_data = NULL;
    }
    ret = true;
cleanup:
    free(param_mem);
    free(param_data);
    return ret;
}

bool cricket_cr_ckp_params(CUDBGAPI cudbgAPI, const char *ckp_dir,
                           cricket_elf_info *elf_info, uint32_t dev,
                           uint32_t sm, uint32_t warp)
{
    CUDBGResult res;
    uint8_t *param_mem = NULL;
    void *param_data = NULL;
    size_t heapsize;
    char heap_suffix[8];
    bool ret = false;
    /* Parameters are the same for all warps so just use warp 0
     * TODO: use first valid warp, because warp 0 may not be in use (is that
     * possible?)
     */
    if ((param_mem = malloc(elf_info->param_size)) == NULL)
        return false;

    res =
        cudbgAPI->readParamMemory(dev, sm, warp, (uint64_t)elf_info->param_addr,
                                  param_mem, (uint32_t)elf_info->param_size);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d):%s", __LINE__,
                cudbgGetErrorString(res));
        goto cleanup;
    }

#ifdef _CRICKET_DEBUG_CR_
    printf("param-mem: ");
    for (int i = 0; i != elf_info->param_size; ++i) {
        printf("%02x ", param_mem[i]);
    }
    printf("\n");
#endif

    if (!cricket_file_store_mem(ckp_dir, CRICKET_DT_PARAM, NULL, param_mem,
                                elf_info->param_size)) {
        printf("error writing param\n");
        goto cleanup;
    }
    cricket_focus_host(0);
    for (int i = 0; i != elf_info->param_num; ++i) {
        //continue;
        if (elf_info->params[i].size != 8)
            continue;

        if (!cricket_heap_memreg_size(
                 *(void **)(param_mem + elf_info->params[i].offset),
                 &heapsize)) {
            printf("cricket-heap: param %u is a 64 bit parameter but does not "
                   "point to an allocated region or an error occured\n",
                   i);
            continue;
        }

        printf("heap param %u: %llx (%lu)\n", i,
               *(void **)(param_mem + elf_info->params[i].offset), heapsize);

        if ((param_data = realloc(param_data, heapsize)) == NULL) {
            goto cleanup;
        }

        res = cudbgAPI->readGlobalMemory(
            *(uint64_t *)(param_mem + elf_info->params[i].offset), param_data,
            heapsize);
        if (res != CUDBG_SUCCESS) {
            fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
                    cudbgGetErrorString(res));
            goto cleanup;
        }

#ifdef _CRICKET_DEBUG_CR_
        printf("param-data for param %u: ", i);
        for (int i = 0; i != heapsize; ++i) {
            printf("%02x ", ((uint8_t *)param_data)[i]);
        }
        printf("\n");
#endif

        sprintf(heap_suffix, "-P%u", elf_info->params[i].index);
        if (!cricket_file_store_mem(ckp_dir, CRICKET_DT_HEAP, heap_suffix,
                                    param_data, heapsize)) {
            printf("cricket error while writing param heap\n");
        }
    }
    ret = true;
cleanup:
    free(param_mem);
    free(param_data);
    return ret;
}

bool cricket_cr_ckp_shared(CUDBGAPI cudbgAPI, const char *ckp_dir,
                           cricket_elf_info *elf_info, uint32_t dev,
                           uint32_t sm, uint32_t warp)
{
    CUDBGResult res;
    uint8_t *shared_mem = NULL;
    char warp_suffix[16];
    bool ret = false;
    if (elf_info->shared_size == 0)
        return true;

    if ((shared_mem = malloc(elf_info->shared_size)) == NULL)
        return false;

    sprintf(warp_suffix, "-D%uS%uW%u", dev, sm, warp);

    res = cudbgAPI->readSharedMemory(dev, sm, warp, 0x0LLU, shared_mem,
                                     (uint32_t)elf_info->shared_size);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d):%s\n", __LINE__,
                cudbgGetErrorString(res));
        goto cleanup;
    }

#ifdef _CRICKET_DEBUG_CR_
    printf("shared-mem (%u): ", elf_info->shared_size);
    for (int i = 0; i != elf_info->shared_size; ++i) {
        printf("%02x ", shared_mem[i]);
    }
    printf("\n");
#endif

    if (!cricket_file_store_mem(ckp_dir, CRICKET_DT_SHARED, warp_suffix,
                                shared_mem, elf_info->shared_size)) {
        printf("error writing param\n");
        goto cleanup;
    }
    ret = true;
cleanup:
    free(shared_mem);
    return ret;
}

bool cricket_cr_rst_shared(CUDBGAPI cudbgAPI, const char *ckp_dir,
                           cricket_elf_info *elf_info, uint32_t dev,
                           uint32_t sm, uint32_t warp)
{
    CUDBGResult res;
    uint8_t *shared_mem = NULL;
    char warp_suffix[16];
    bool ret = false;
    if (elf_info->shared_size == 0)
        return true;

    if ((shared_mem = malloc(elf_info->shared_size)) == NULL)
        return false;

    sprintf(warp_suffix, "-D%uS%uW%u", dev, sm, warp);

    if (!cricket_file_read_mem(ckp_dir, CRICKET_DT_SHARED, warp_suffix,
                               shared_mem, elf_info->shared_size)) {
        printf("error reading shared\n");
        goto cleanup;
    }

#ifdef _CRICKET_DEBUG_CR_
    printf("shared-mem (%u): ", elf_info->shared_size);
    for (int i = 0; i != elf_info->shared_size; ++i) {
        printf("%02x ", shared_mem[i]);
    }
    printf("\n");
#endif

    res = cudbgAPI->writeSharedMemory(dev, sm, warp, 0x0LLU, shared_mem,
                                      (uint32_t)elf_info->shared_size);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d):%s", __LINE__,
                cudbgGetErrorString(res));
        goto cleanup;
    }

    ret = true;
cleanup:
    free(shared_mem);
    return ret;
}

bool cricket_cr_rst_globals(CUDBGAPI cudbgAPI, const char *ckp_dir)
{
    CUDBGResult res;
    uint8_t *globals_mem = NULL;
    void *globals_data = NULL;
    size_t heapsize;
    char *heap_suffix = NULL;
    bool ret = false;
    size_t i;
    cricket_global_var *globals;
    size_t globals_num;
    size_t globals_mem_size = 0;
    size_t offset = 0;

    /* Globals are the same for all warps and SMs so just use warp 0
     * TODO: use first valid warp, because warp 0 may not be in use (is that
     * possible?)
     */
    if (!cricket_elf_get_global_vars_info(&globals, &globals_num)) {
        printf("cricket-cr: error getting global variable info\n");
        return false;
    }

    for (i = 0; i < globals_num; ++i) {
        globals_mem_size += globals[i].size;
    }

    if ((globals_mem = malloc(globals_mem_size)) == NULL) {
        return false;
    }

    if (!cricket_file_read_mem(ckp_dir, CRICKET_DT_GLOBALS, NULL, globals_mem,
                               globals_mem_size)) {
        fprintf(stderr, "cricket-cr: error while reading globals\n");
        goto cleanup;
    }

#ifdef _CRICKET_DEBUG_CR_
    printf("globals-mem: ");
    for (int i = 0; i != globals_mem_size; ++i) {
        printf("%02x ", globals_mem[i]);
    }
    printf("\n");
#endif

    for (i = 0; i < globals_num; ++i) {
        res = cudbgAPI->writeGlobalMemory(
            globals[i].address, globals_mem + offset, globals[i].size);
        if (res != CUDBG_SUCCESS) {
            fprintf(stderr, "cricket-cr (%d):%s", __LINE__,
                    cudbgGetErrorString(res));
            goto cleanup;
        }
        offset += globals[i].size;
    }

    offset = 0;
    for (i = 0; i != globals_num; ++i) {
        if (globals[i].size != 8)
            continue;

        asprintf(&heap_suffix, "-G%s", globals[i].symbol);

        if (!cricket_file_exists(ckp_dir, CRICKET_DT_HEAP, heap_suffix)) {
            printf("cricket-cr: no checkpoint file for global variable %s\n",
                   globals[i].symbol);
            free(heap_suffix);
            heap_suffix = NULL;
            continue;
        }
        globals_data = NULL;
        if (!cricket_file_read_mem_size(ckp_dir, CRICKET_DT_HEAP, heap_suffix,
                                        &globals_data, 0, &heapsize)) {
            printf("cricket error while writing globals heap\n");
            free(heap_suffix);
            heap_suffix = NULL;
            goto cleanup;
        }
        free(heap_suffix);
        heap_suffix = NULL;

#ifdef _CRICKET_DEBUG_CR_
        printf("heap global %u: %llx (%u)\n", i,
               *(void **)(globals_mem + offset), heapsize);
        printf("globals-data for global variable %s: ", globals[i].symbol);
        for (int i = 0; i != heapsize; ++i) {
            printf("%02x ", ((uint8_t *)globals_data)[i]);
        }
        printf("\n");
#endif

        res = cudbgAPI->writeGlobalMemory(*(uint64_t *)(globals_mem + offset),
                                          globals_data, heapsize);
        if (res != CUDBG_SUCCESS) {
            fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
                    cudbgGetErrorString(res));
            goto cleanup;
        }
        free(globals_data);
        globals_data = NULL;

        offset += globals[i].size;
    }
    ret = true;
cleanup:
    free(globals_mem);
    free(globals_data);
    return ret;
}

bool cricket_cr_ckp_globals(CUDBGAPI cudbgAPI, const char *ckp_dir)
{
    CUDBGResult res;
    uint8_t *globals_mem = NULL;
    void *globals_data = NULL;
    size_t heapsize;
    char *heap_suffix = NULL;
    bool ret = false;
    size_t i;
    cricket_global_var *globals;
    size_t globals_num;
    size_t globals_mem_size = 0;
    size_t offset = 0;

    /* Globals are the same for all warps and SMs so just use warp 0
     * TODO: use first valid warp, because warp 0 may not be in use (is that
     * possible?)
     */
    if (!cricket_elf_get_global_vars_info(&globals, &globals_num)) {
        printf("cricket-cr: error getting global variable info\n");
        return false;
    }

    for (i = 0; i < globals_num; ++i) {
        globals_mem_size += globals[i].size;
    }

    if ((globals_mem = malloc(globals_mem_size)) == NULL) {
        return false;
    }

    for (i = 0; i < globals_num; ++i) {
        res = cudbgAPI->readGlobalMemory(globals[i].address,
                                         globals_mem + offset, globals[i].size);
        if (res != CUDBG_SUCCESS) {
            fprintf(stderr, "cricket-cr (%d):%s", __LINE__,
                    cudbgGetErrorString(res));
            goto cleanup;
        }
        offset += globals[i].size;
    }

#ifdef _CRICKET_DEBUG_CR_
    printf("globals-mem: ");
    for (int i = 0; i != globals_mem_size; ++i) {
        printf("%02x ", globals_mem[i]);
    }
    printf("\n");
#endif

    if (!cricket_file_store_mem(ckp_dir, CRICKET_DT_GLOBALS, NULL, globals_mem,
                                globals_mem_size)) {
        fprintf(stderr, "cricket-cr: error writing globals\n");
        goto cleanup;
    }

    offset = 0;
    cricket_focus_host(0);
    for (i = 0; i != globals_num; ++i, offset += globals[i].size) {
        //continue;
        if (globals[i].size != 8)
            continue;

        if (!cricket_heap_memreg_size(*(void **)(globals_mem + offset),
                                      &heapsize)) {
            printf("cricket-heap: global variable %s has a size of 64 bit but "
                   "does not point to an allocated region or an error "
                   "occured\n",
                   globals[i].symbol);
            continue;
        }
        printf("heap param %u: %llx (%lu)\n", i,
               *(void **)(globals_mem + offset), heapsize);

        if ((globals_data = realloc(globals_data, heapsize)) == NULL) {
            goto cleanup;
        }

        res = cudbgAPI->readGlobalMemory(*(uint64_t *)(globals_mem + offset),
                                         globals_data, heapsize);
        if (res != CUDBG_SUCCESS) {
            fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
                    cudbgGetErrorString(res));
            goto cleanup;
        }

#ifdef _CRICKET_DEBUG_CR_
        printf("global-data for global variable %s: ", globals[i].symbol);
        for (int i = 0; i != heapsize; ++i) {
            printf("%02x ", ((uint8_t *)globals_data)[i]);
        }
        printf("\n");
#endif
        asprintf(&heap_suffix, "-G%s", globals[i].symbol);
        if (!cricket_file_store_mem(ckp_dir, CRICKET_DT_HEAP, heap_suffix,
                                    globals_data, heapsize)) {
            printf("cricket error while writing param heap\n");
        }
        free(heap_suffix);
        heap_suffix = NULL;
    }
    ret = true;
cleanup:
    free(globals_mem);
    free(globals_data);
    return ret;
}
