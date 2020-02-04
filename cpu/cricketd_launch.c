#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include <config.h>
#include <bfd.h>
#include <libbfd.h>
#include <elf-bfd.h>

#include "cricketd_launch.h"

#define CRICKETD_LAUNCH_FATBIN_SEG ".nvFatBinSegment"

extern void** __cudaRegisterFatBinary(
  void *fatCubin
);
extern void __cudaRegisterFunction(
  void **fatCubinHandle, const char *hostFun, char *deviceFun,
  const char *deviceName, int thread_limit, void *tid,
  void *bid, void *bDim, void *gDim, int *wSize
);

static int cricket_elf_print_symtab(bfd *abfd)
{
    size_t symtab_size, symtab_length;
    asymbol **symtab;

    if ((symtab_size = bfd_get_symtab_upper_bound(abfd)) == -1) {
        fprintf(stderr, "cricket-elf: bfd_get_symtab_upper_bound failed\n");
        return 0;
    }

    printf("symtab size: %lu\n", symtab_size);

    if ((symtab = (asymbol **)malloc(symtab_size)) == NULL) {
        fprintf(stderr, "cricket-elf: malloc symtab failed\n");
        return 0;
    }

    if ((symtab_length = bfd_canonicalize_symtab(abfd, symtab)) == 0) {
        printf("symtab empty...\n");
    } else {
        printf("%lu symtab entries\n", symtab_length);
    }

    for (int i = 0; i < symtab_length; ++i) {
        printf("%d: %s: %lx (%lx, %lx)\n", i, bfd_asymbol_name(symtab[i]),
               bfd_asymbol_value(symtab[i]), symtab[i]->section->size, symtab[i]->section->rawsize);
    }
    free(symtab);
    return 1;
}

int cricketd_launch_load_elf(const char *filename, void **fatbin, size_t *fatbin_size)
{
    bfd *hostbfd = NULL;
    bfd *cudabfd = NULL;
    asection *section;
    size_t fatbin_pos;
    FILE *hostbfd_fd = NULL;
    FILE *cudabfd_fd = NULL;
    *fatbin = NULL;
    int ret = 0;

    if (filename == NULL) {
        fprintf(stderr, "cricket_elf (%d): filename is NULL\n", __LINE__);
        return ret;
    }

    bfd_init();

    if ((hostbfd_fd = fopen(filename, "rb")) == NULL) {
        fprintf(stderr, "cricket-elf (%d): fopen failed\n", __LINE__);
        return ret;
    }

    if ((hostbfd = bfd_openstreamr(filename, NULL, hostbfd_fd)) == NULL) {
        fprintf(stderr, "cricket-elf (%d): bfd_openr failed on %s\n", __LINE__,
                filename);
        fclose(hostbfd_fd);
        goto cleanup;
    }

    if (!bfd_check_format(hostbfd, bfd_object)) {
        fprintf(stderr, "cricket-elf (%d): %s has wrong bfd format\n", __LINE__,
                filename);
        goto cleanup;
    }

    section = bfd_get_section_by_name(hostbfd, CRICKETD_LAUNCH_FATBIN_SEG);
    if (section == NULL) {
        fprintf(stderr, "cricket-elf (%d): fatbin section %s not found\n",
                __LINE__, CRICKETD_LAUNCH_FATBIN_SEG);
        goto cleanup;
    }

    printf("name: %s, index: %d, size %lx, pos:%lx\n", section->name,
           section->index, section->size, (void *)section->filepos);
    cricket_elf_print_symtab(hostbfd);

    fatbin_pos = section->filepos;
    *fatbin_size = section->size;

    // obtain file size:
    fseek (hostbfd_fd, 0, SEEK_END);
    *fatbin_size = ftell (hostbfd_fd);
    rewind (hostbfd_fd);

    fatbin_pos = 0;

    if ((*fatbin = malloc(*fatbin_size)) == NULL) {
        goto cleanup;
    }
    if (fseek(hostbfd_fd, fatbin_pos, SEEK_SET) != 0) {
        fprintf(stderr, "cricket-elf: fseek failed\n");
        goto cleanup;
    }
    if (fread(*fatbin, *fatbin_size, 1, hostbfd_fd) != 1) {
        fprintf(stderr, "cricket-elf: fread failed\n");
        goto cleanup;
    }

    struct __fatCubin fat = *((struct __fatCubin*)((*fatbin) + section->filepos));
    for (int i=section->filepos; i < section->filepos+section->size; ++i) {
        printf("%02x ", ((char*)(*fatbin))[i]);
    }
    printf("\n");

    printf("magic: %x, seq: %x, text: %lx, data: %lx, ptr: %lx, ptr2: %lx, zero: %lx\n",
           fat.magic, fat.seq, fat.text, fat.data, fat.ptr, fat.ptr2, fat.zero);

    fat.text += (uint64_t)*fatbin;
    fat.data += (uint64_t)*fatbin;
    fat.ptr2 += (uint64_t)*fatbin;

    printf("magic: %x, seq: %x, text: %lx, data: %lx, ptr: %lx, ptr2: %lx, zero: %lx\n",
           fat.magic, fat.seq, fat.text, fat.data, fat.ptr, fat.ptr2, fat.zero);


    void **fatCubinHandle = __cudaRegisterFatBinary(&fat);
    printf("%p\n", fatCubinHandle);


    char *fun_name = "_Z6kernelPtS_S_csix";

    section = bfd_get_section_by_name(hostbfd, fun_name);
    if (section == NULL) {
        fprintf(stderr, "cricket-elf (%d): fatbin section %s not found\n",
                __LINE__, fun_name);
        goto cleanup;
    }

    printf("function %s is at %p and %p bytes long\n", section->filepos, section->size);

    __cudaRegisterFunction(fatCubinHandle, (void*)(0x40168d+*fatbin), fun_name, fun_name,                            -1, NULL, NULL, NULL, NULL, NULL);
    /*if ((cudabfd_fd = fmemopen(fatbin, fatbin_size, "rb")) == NULL) {
        fprintf(stderr, "cricket-elf (%d): fmemopen failed\n", __LINE__);
        goto cleanup;
    }

    if ((cudabfd = bfd_openstreamr(filename, NULL, cudabfd_fd)) == NULL) {
        fprintf(stderr, "cricket-elf: bfd_openstreamr failed\n");
        fclose(cudabfd_fd);
        goto cleanup;
    }

    if (!bfd_check_format(cudabfd, bfd_object)) {
        fprintf(stderr, "cricket-elf: wrong bfd format\n");
        goto cleanup;
    }

#ifdef _CRICKET_ELF_DEBUG_
    cricket_elf_print_mem(fatbin, 0x160);
    cricket_elf_print_symtab(cudabfd);
#endif

    for (section = cudabfd->sections; section != NULL;
         section = section->next) {
        if (strncmp(section->name, CRICKET_ELF_TEXT_PREFIX, text_prefixlen) !=
            0) {
            continue;
        }

        printf("name: %s, index: %d, size %lx, pos:%p\n", section->name,
               section->index, section->size, (void *)section->filepos);

    }
*/
    ret = 1;
cleanup:
    if (ret != 1) free(*fatbin);
    /*if (cudabfd != NULL)
        bfd_close(cudabfd);
        */
    if (hostbfd != NULL)
        bfd_close(hostbfd);
    return ret;
}

