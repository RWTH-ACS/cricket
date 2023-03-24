#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include "cpu-common.h"
#include "log.h"
#include "cpu-elf.h"
#include "cpu-utils.h"

#include "bfd_extracts.h"

#define uint16_t unsigned short
#define CRICKET_ELF_NV_INFO_PREFIX ".nv.info"
#define CRICKET_ELF_NV_SHARED_PREFIX ".nv.shared."
#define CRICKET_ELF_NV_TEXT_PREFIX ".nv.text."
#define CRICKET_ELF_TEXT_PREFIX ".text."

#define CRICKET_ELF_FATBIN ".nv_fatbin"
#define CRICKET_ELF_REGFUN "_ZL24__sti____cudaRegisterAllv"

#define FATBIN_STRUCT_MAGIC 0x466243b1
#define FATBIN_TEXT_MAGIC   0xBA55ED50

struct  __attribute__((__packed__)) fat_elf_header
{
    uint32_t magic;
    uint16_t version;
    uint16_t header_size;
    uint64_t fat_size;
};
struct  __attribute__((__packed__)) fat_text_header
{
    uint16_t kind;
    uint16_t unknown1;
    uint32_t header_size;
    uint64_t fatbin_size;
    uint64_t some_offset; //Compression related information
    uint16_t minor;
    uint16_t major;
    uint32_t arch;
    uint32_t obj_name_offset;
    uint32_t obj_name_len;
    uint64_t flags;
    uint64_t zero;
    uint64_t unknown2;
};

#define FATBIN_FLAG_64BIT     0x0000000000000001LL
#define FATBIN_FLAG_DEBUG     0x0000000000000002LL
#define FATBIN_FLAG_LINUX     0x0000000000000010LL
#define FATBIN_FLAG_COMPRESS  0x0000000000002000LL

static int flag_to_str(char** str, uint64_t flag)
{
    return asprintf(str, "64Bit: %s, Debug: %s, Linux: %s, Compress %s",
        (flag & FATBIN_FLAG_64BIT) ? "yes" : "no",
        (flag & FATBIN_FLAG_DEBUG) ? "yes" : "no",
        (flag & FATBIN_FLAG_LINUX) ? "yes" : "no",
        (flag & FATBIN_FLAG_COMPRESS) ? "yes" : "no");
}

static int fat_header_decode(void *fat, 
                                       struct fat_elf_header **fat_elf_header,
                                       struct fat_text_header **fat_text_header,
                                       void **fat_text_body_ptr)
{
    struct fat_elf_header* feh;
    struct fat_text_header* fth;
    void *fat_ptr = NULL;
    void *fat_text_header_ptr = NULL;

    if (fat == NULL || fat_elf_header == NULL || fat_text_header == NULL || fat_text_body_ptr == NULL) {
        LOGE(LOG_ERROR, "at least one parameter is NULL");
        return -1;
    }

    feh = (struct fat_elf_header*)fat;
    if (feh->magic != FATBIN_TEXT_MAGIC) {
        LOGE(LOG_ERROR, "fatbin text magic number is wrong. Got %x, expected %x.", *((uint32_t*)feh), FATBIN_TEXT_MAGIC);
        return -1;
    }
    LOGE(LOG_DBG(1), "fat_elf_header: magic: %x, version: %d, header_size: %p, fat_size: %p",
        feh->magic, feh->version, feh->header_size, feh->fat_size);

    if (feh->version != 1 || feh->header_size != sizeof(struct fat_elf_header)) {
        LOGE(LOG_ERROR, "fatbin text version is wrong or header size is inconsistent.\
            This is a sanity check to avoid reading a new fatbinary format");
        return -1;
    }
    fat_ptr = fat_text_header_ptr = (void*)feh + feh->header_size;

    fth = (struct fat_text_header*)(fat_text_header_ptr);
    LOGE(LOG_DBG(1), "fat_text_header: fatbin_kind: %#x, header_size %#x, fatbin_size %#x, some_offset %#x.\
        minor %#x, major %#x, arch %d, flags %#x",
        fth->kind,
        fth->header_size,
        fth->fatbin_size,
        fth->some_offset,
        fth->minor,
        fth->major,
        fth->arch,
        fth->flags);
    LOGE(LOG_DBG(1), "unknown fields: unknown1: %#x, unknown2: %#x, zeros: %#x",
        fth->unknown1,
        fth->unknown2,
        fth->zero);
    fat_ptr += sizeof(struct fat_header);
    *fat_text_body_ptr = fat_text_header_ptr + fth->header_size;
    if (fth->flags & FATBIN_FLAG_DEBUG) {
        *fat_text_body_ptr += 1;
    }

    char *flag_str = NULL;
    flag_to_str(&flag_str, fth->flags);
    LOGE(LOG_DBG(1), "Fatbin flags: %s", flag_str);
    free(flag_str);

    if(fth->obj_name_offset != 0) {
        if (((char*)fat_text_header_ptr)[fth->obj_name_offset + fth->obj_name_len] != '\0') {
            LOGE(LOG_DEBUG, "Fatbin object name is not null terminated");
        } else {
            char *obj_name = (char*)fat_text_header_ptr + fth->obj_name_offset;
            LOGE(LOG_DEBUG, "Fatbin object name: %s (len:%#x)", obj_name, fth->obj_name_len);
        }
        fat_ptr += fth->obj_name_len+1;
    }
    *fat_elf_header = feh;
    *fat_text_header = fth;
    return 0;
}

int elf_get_fatbin_info(struct fat_header *fatbin, list *kernel_infos, void** fatbin_mem, unsigned* fatbin_size)
{
    struct fat_elf_header* fat_elf_header;
    struct fat_text_header* fat_text_header;
    void *fat_ptr = NULL;
    void *fat_text_body_ptr = NULL;
    unsigned fatbin_total_size = 0;
    if (fatbin == NULL || fatbin_mem == NULL || fatbin_size == NULL) {
        LOGE(LOG_ERROR, "at least one parameter is NULL");
        return -1;
    }
    if (fatbin->magic != FATBIN_STRUCT_MAGIC) {
        LOGE(LOG_ERROR, "fatbin struct magic number is wrong. Got %llx, expected %llx.", fatbin->magic, FATBIN_STRUCT_MAGIC);
        return -1;
    }
    LOG(LOG_DBG(1), "Fatbin: magic: %x, version: %x, text: %lx, data: %lx, ptr: %lx, ptr2: %lx, zero: %lx",
           fatbin->magic, fatbin->version, fatbin->text, fatbin->data, fatbin->unknown, fatbin->text2, fatbin->zero);

    if (fat_header_decode((void*)fatbin->text, &fat_elf_header, &fat_text_header, &fat_text_body_ptr) != 0) {
        LOGE(LOG_ERROR, "fatbin header decode failed");
        return -1;
    }


    fatbin_total_size = fat_elf_header->header_size + fat_elf_header->fat_size;

    // for (int i=0; i<64; i++) {
    //     printf("%02x ", ((uint8_t*)fat_text_body_ptr)[i]);
    // }
    // printf("\n");

    if (elf_parameter_info(kernel_infos, fat_text_body_ptr, fat_elf_header->fat_size) != 0) {
        LOGE(LOG_ERROR, "error getting symbol table");
        return -1;
    }

    if (fat_header_decode((void*)fatbin->text2, &fat_elf_header, &fat_text_header, &fat_text_body_ptr) != 0) {
        LOGE(LOG_ERROR, "fatbin header decode failed");
        return -1;
    }
    fatbin_total_size += fat_elf_header->header_size + fat_elf_header->fat_size;

    // if (cricketd_utils_symtab(fat_text_body_ptr, fat_elf_header->fat_size) == NULL) {
    //     LOGE(LOG_ERROR, "error getting symbol table");
    //     return -1;
    // }
    fat_ptr = (void*)fatbin->data;

    // for (int i=0; i<64; i++) {
    //     printf("%02x ", ((uint8_t*)fatbin->text)[i]);
    // }
    // printf("\n");

    *fatbin_mem = (void*)fatbin->text;
    *fatbin_size = fatbin_total_size;
    return 0;
}

size_t cudabfd_size = 0;
int (*orig_cudabfd_stat)(struct bfd *abfd, struct stat* sb);
int cudabfd_stat(struct bfd *bfd, struct stat *sb)
{
    //int ret = orig_cudabfd_stat(bfd, sb);
    sb->st_size = cudabfd_size;
    return 0;
}

static void print_sections(asection *sections)
{
    for (asection *section = sections; section != NULL; section = section->next) {
        printf("section: %s (len: %#x)\n", section->name, section->size);
    }
}

static void print_hexmem(void *mem, size_t len)
{
    for (int i=0; i<len; i++) {
        printf("%02x ", ((uint8_t*)mem)[i]);
    }
    printf("\n");
}

struct symtab {
    asymbol **symtab;
    size_t symtab_size;
    size_t symtab_length;
};

static int symtab_init(bfd *bfd, struct symtab *st)
{
    if (st == NULL || bfd == NULL) {
        LOGE(LOG_ERROR, "at least one parameter is NULL");
        return -1;
    }

    if (memset(st, 0, sizeof(struct symtab)) == NULL) {
        LOGE(LOG_ERROR, "memset failed");
        return -1;
    }

    if ((st->symtab_size = bfd_get_symtab_upper_bound(bfd)) == -1) {
        LOGE(LOG_ERROR, "bfd_get_symtab_upper_bound failed");
        return -1;
    }

    if ((st->symtab = (asymbol **)malloc(st->symtab_size)) == NULL) {
        LOGE(LOG_ERROR, "malloc symtab failed");
        return -1;
    }

    if ((st->symtab_length = bfd_canonicalize_symtab(bfd, st->symtab)) == 0) {
        LOG(LOG_WARNING, "symtab is empty...");
    } else {
        LOGE(LOG_DBG(1), "%lu symtab entries", st->symtab_length);
    }
    return 0;
}

static void symtab_free(struct symtab* st)
{
    if (st == NULL) {
        return;
    }
    free(st->symtab);
    memset(st, 0, sizeof(struct symtab));
}

static int symtab_symbol_at(struct symtab* st, size_t index, const char** sym)
{
    if (st == NULL || sym == NULL) {
        LOGE(LOG_ERROR, "at least one parameter is NULL");
        return -1;
    }

    if (index >= st->symtab_length+1 || index == 0) {
        LOGE(LOG_ERROR, "index out of bounds");
        return -1;
    }
    // The first entry of any symbol table is for undefined symbols and is always zero.
    // Libbfd ignores this entry, but readelf does not so there is a difference of one
    // between libbfd indices and those referenced by the .nv.info sections.
    *sym = bfd_asymbol_name(st->symtab[index-1]);
    return 0;
}

static void symtab_print(struct symtab* st)
{
    const char* sym;
    for (int i = 1; i < st->symtab_length+1; ++i) {
        symtab_symbol_at(st, i, &sym);
        printf("%#x: name: %s\n", i, sym);
    }
}

#define EIATTR_PARAM_CBANK              0xa
#define EIATTR_EXTERNS                  0xf
#define EIATTR_FRAME_SIZE               0x11
#define EIATTR_MIN_STACK_SIZE           0x12
#define EIATTR_KPARAM_INFO              0x17
#define EIATTR_CBANK_PARAM_SIZE         0x19
#define EIATTR_MAX_REG_COUNT            0x1b
#define EIATTR_EXIT_INSTR_OFFSETS       0x1c
#define EIATTR_S2RCTAID_INSTR_OFFSETS   0x1d
#define EIATTR_CRS_STACK_SIZE           0x1e
#define EIATTR_SW1850030_WAR            0x2a
#define EIATTR_REGCOUNT                 0x2f
#define EIATTR_SW2393858_WAR            0x30
#define EIATTR_INDIRECT_BRANCH_TARGETS  0x34
#define EIATTR_CUDA_API_VERSION         0x37

#define EIFMT_NVAL                      0x1
#define EIFMT_HVAL                      0x3
#define EIFMT_SVAL                      0x4

static int get_parm_for_kernel(bfd *bfd,  kernel_info_t *kernel, void* memory, size_t memsize)
{
    struct __attribute__((__packed__)) nv_info_kernel_entry {
        uint8_t format;
        uint8_t attribute;
        uint16_t values_size;
        uint32_t values;
    };
    struct __attribute__((__packed__)) nv_info_kparam_info {
        uint32_t index;
        uint16_t ordinal;
        uint16_t offset;
        uint16_t unknown : 12;
        uint8_t  cbank : 6;
        uint16_t size : 14;
        // missing are "space" (possible padding info?), and "Pointee's logAlignment"
        // these were always 0 in the kernels I tested
    };
    asection *section = NULL;
    int ret = -1;
    char *section_name = NULL;

    if (bfd == NULL || kernel == NULL || kernel->name == NULL || memory == NULL) {
        LOGE(LOG_ERROR, "at least one parameter is NULL");
        return ret;
    }
    kernel->param_num = 0;
    kernel->param_offsets = NULL;
    kernel->param_sizes = NULL;

    if (asprintf(&section_name, ".nv.info.%s", kernel->name) == -1) {
        LOGE(LOG_ERROR, "asprintf failed");
        return ret;
    }

    if ((section = bfd_get_section_by_name(bfd, section_name))== NULL) {
        LOGE(LOG_ERROR, "%s section not found", section_name);
        goto cleanup;
    }

    LOGE(LOG_DBG(1), "name: %s, index: %d, size 0x%lx, pos:%p", section->name,
        section->index, section->size, (void *)section->filepos);

    //print_hexmem(memory+section->filepos, section->size);

    size_t secpos=0;
    int i=0;
    while (secpos < section->size) {
        struct nv_info_kernel_entry *entry = (struct nv_info_kernel_entry*)(memory+section->filepos+secpos);
        // printf("entry %d: format: %#x, attr: %#x, ", i++, entry->format, entry->attribute);
        if (entry->format == EIFMT_SVAL && entry->attribute == EIATTR_KPARAM_INFO) {
            if (entry->values_size != 0xc) {
                LOGE(LOG_ERROR, "EIATTR_KPARAM_INFO values size has not the expected value of 0xc");
                goto cleanup;
            }
            struct nv_info_kparam_info *kparam = (struct nv_info_kparam_info*)&entry->values;
            // printf("kparam: index: %#x, ordinal: %#x, offset: %#x, unknown: %#0x, cbank: %#0x, size: %#0x\n",
            //     kparam->index, kparam->ordinal, kparam->offset, kparam->unknown, kparam->cbank, kparam->size);
            LOGE(LOG_DEBUG, "param %d: offset: %#x, size: %#x", kparam->ordinal, kparam->offset, kparam->size);
            if (kparam->ordinal >= kernel->param_num) {
                kernel->param_offsets = realloc(kernel->param_offsets,
                                              (kparam->ordinal+1)*sizeof(uint16_t));
                kernel->param_sizes = realloc(kernel->param_sizes,
                                            (kparam->ordinal+1)*sizeof(uint16_t));
                kernel->param_num = kparam->ordinal+1;
            }
            kernel->param_offsets[kparam->ordinal] = kparam->offset;
            kernel->param_sizes[kparam->ordinal] = kparam->size;
            secpos += sizeof(struct nv_info_kernel_entry) + entry->values_size-4;
        } else if (entry->format == EIFMT_HVAL && entry->attribute == EIATTR_CBANK_PARAM_SIZE) {
            kernel->param_size = entry->values_size;
            LOGE(LOG_DEBUG, "cbank_param_size: %#0x", entry->values_size);
            secpos += sizeof(struct nv_info_kernel_entry)-4;
        } else if (entry->format == EIFMT_HVAL) {
            // printf("hval: %#x(%d)\n", entry->values_size, entry->values_size);
            secpos += sizeof(struct nv_info_kernel_entry)-4;
        } else if (entry->format == EIFMT_SVAL) {
            // printf("sval_size: %#x ", entry->values_size);
            // for (int j=0; j*sizeof(uint32_t) < entry->values_size; j++) {
            //     printf("val%d: %#x(%d) ", j, (&entry->values)[j], (&entry->values)[j]);
            // }
            // printf("\n");
            secpos += sizeof(struct nv_info_kernel_entry) + entry->values_size-4;
        } else if (entry->format == EIFMT_NVAL) {
            // printf("nval\n");
            secpos += sizeof(struct nv_info_kernel_entry)-4;
        } else {
            LOGE(LOG_WARNING, "unknown format: %#x", entry->format);
            secpos += sizeof(struct nv_info_kernel_entry)-4;
        }
    }
    // printf("remaining: %d\n", section->size % sizeof(struct nv_info_kernel_entry));
    ret = 0;
 cleanup:
    free(section_name);
    return ret;
}

#define ELF_DUMP_TO_FILE 1

int elf_parameter_info(list *kernel_infos, void* memory, size_t memsize)
{
    struct __attribute__((__packed__)) nv_info_entry{
        uint8_t format;
        uint8_t attribute;
        uint16_t values_size;
        uint32_t kernel_id;
        uint32_t value;
    };

    bfd *bfd = NULL;
    FILE *fd = NULL;
    asection *section = NULL;
    int ret = -1;
    struct symtab symtab = {0};
    char path[256];
    struct bfd_iovec *iovec = NULL;
    const struct bfd_iovec *orig_iovec = NULL;

    kernel_info_t *ki = NULL;

    if (memory == NULL || memsize == 0) {
        LOGE(LOG_ERROR, "memory was NULL or memsize was 0");
        return -1;
    }

#ifdef ELF_DUMP_TO_FILE
    FILE* fd2 = fopen("/tmp/cricket-elf-dump", "wb");
    fwrite(memory, memsize, 1, fd2);
    fclose(fd2);
#endif

    if ((fd = fmemopen(memory, memsize, "rb")) == NULL) {
        LOGE(LOG_ERROR, "fmemopen failed");
        goto cleanup;
    }

    bfd_init();

    if ((bfd = bfd_openstreamr("", "elf64-little", fd)) == NULL) {
        LOGE(LOG_ERROR, "bfd_openr failed");
        goto cleanup;
    }

    //We change the iovec of cudabfd so we can report the correct filesize
    //because in-memory files always report a file size of 0, which creates 
    //problems elsewhere
    cudabfd_size = memsize;
    orig_cudabfd_stat = bfd->iovec->bstat;
    orig_iovec = bfd->iovec;
    iovec = (struct bfd_iovec*)malloc(sizeof(struct bfd_iovec));
    memcpy(iovec, bfd->iovec, sizeof(struct bfd_iovec));
    iovec->bstat = cudabfd_stat;
    bfd->iovec = iovec;

    if (!bfd_check_format(bfd, bfd_object)) {
        LOGE(LOG_ERROR, "bfd has wrong format");
        goto cleanup;
    }
    // print_sections(bfd->sections);

    if  (symtab_init(bfd, &symtab) != 0) {
        LOGE(LOG_ERROR, "symtab_init failed");
        goto cleanup;
    }
    // symtab_print(&symtab);

    section = bfd_get_section_by_name(bfd, ".nv.info");
    if (section == NULL) {
        LOGE(LOG_ERROR, ".nv.info section not found");
        goto cleanup;
    }

    LOGE(LOG_DBG(1), "name: %s, index: %d, size 0x%lx, pos:%p", section->name,
        section->index, section->size, (void *)section->filepos);
    //print_hexmem(memory+section->filepos, section->size); 
    int i = 0;
    const char *kernel_str;
    for (size_t secpos=0; secpos < section->size; secpos += sizeof(struct nv_info_entry)) {
        struct nv_info_entry *entry = (struct nv_info_entry*)(memory+section->filepos+secpos);
        if (entry->values_size != 8) {
            LOGE(LOG_ERROR, "unexpected values_size: %#x", entry->values_size);
            continue;
        }
        // printf("%d: format: %#x, attr: %#x, values_size: %#x kernel: %#x, sval: %#x(%d)\n", 
        //         i++, entry->format, entry->attribute, entry->values_size, entry->kernel_id, 
        //         entry->value, entry->value);
        if (entry->attribute != EIATTR_FRAME_SIZE) {
            continue;
        }
        if (symtab_symbol_at(&symtab, entry->kernel_id, &kernel_str) != 0) {
            LOGE(LOG_ERROR, "symtab_symbol_at failed for entry %d", i);
            continue;
        }
        if (utils_search_info(kernel_infos, kernel_str) != NULL) {
            continue;
        }

        LOGE(LOG_DEBUG, "found new kernel: %s (symbol table id: %#x)", kernel_str, entry->kernel_id);

        if (list_append(kernel_infos, (void**)&ki) != 0) {
            LOGE(LOG_ERROR, "error on appending to list");
            goto cleanup;
        }

        size_t buflen = strlen(kernel_str)+1;
        if ((ki->name = malloc(buflen)) == NULL) {
            LOGE(LOG_ERROR, "malloc failed");
            goto cleanup;
        }
        if (strncpy(ki->name, kernel_str, buflen) != ki->name) {
            LOGE(LOG_ERROR, "strncpy failed");
            goto cleanup;
        }

        if (get_parm_for_kernel(bfd, ki, memory, memsize) != 0) {
            LOGE(LOG_ERROR, "get_parm_for_kernel failed for kernel %s", kernel_str);
            goto cleanup;
        }
    }

    ret = 0;
 cleanup:
    free(iovec);
    if (fd != NULL)
        fclose(fd);
    symtab_free(&symtab);
    if (bfd != NULL) {
        // Also closes fd
        bfd_close(bfd);
    }
    return ret;
}


void* elf_symbol_address(const char* file, char *symbol)
{
    bfd *hostbfd = NULL;
    asection *section;
    FILE *hostbfd_fd = NULL;
    void *ret = NULL;
    size_t symtab_size, symtab_length;
    asymbol **symtab = NULL;
    char path[256];
    size_t length;
    const char self[] = "/proc/self/exe";
    if (file == NULL) {
        file = self;
    }


    bfd_init();

    length = readlink(file, path, sizeof(path));

    /* Catch some errors: */
    if (length < 0) {
        LOGE(LOG_WARNING, "error resolving symlink %s.", file);
    } else if (length >= 256) {
        LOGE(LOG_WARNING, "path was too long and was truncated.");
    } else {
        path[length] = '\0';
        LOG(LOG_DEBUG, "opening '%s'", path);
    }

    if ((hostbfd_fd = fopen(file, "rb")) == NULL) {
        LOGE(LOG_ERROR, "fopen failed");
        return NULL;
    }

    if ((hostbfd = bfd_openstreamr(file, NULL, hostbfd_fd)) == NULL) {
        LOGE(LOG_ERROR, "bfd_openr failed on %s",
             file);
        fclose(hostbfd_fd);
        goto cleanup;
    }

    if (!bfd_check_format(hostbfd, bfd_object)) {
        LOGE(LOG_ERROR, "%s has wrong bfd format",
             file);
        goto cleanup;
    }

    if ((symtab_size = bfd_get_symtab_upper_bound(hostbfd)) == -1) {
        LOGE(LOG_ERROR, "bfd_get_symtab_upper_bound failed");
        return NULL;
    }

    if ((symtab = (asymbol **)malloc(symtab_size)) == NULL) {
        LOGE(LOG_ERROR, "malloc symtab failed");
        return NULL;
    }

    if ((symtab_length = bfd_canonicalize_symtab(hostbfd, symtab)) == 0) {
        LOG(LOG_WARNING, "symtab is empty...");
    } else {
        //printf("%lu symtab entries\n", symtab_length);
    }

    for (int i = 0; i < symtab_length; ++i) {
        if (strcmp(bfd_asymbol_name(symtab[i]), CRICKET_ELF_REGFUN) == 0) {
            ret = (void*)bfd_asymbol_value(symtab[i]);
            break;
        }
        //printf("%d: %s: %lx\n", i, bfd_asymbol_name(symtab[i]),
        //       bfd_asymbol_value(symtab[i]));
    }


 cleanup:
    free(symtab);
    if (hostbfd != NULL)
        bfd_close(hostbfd);
    return ret;
}