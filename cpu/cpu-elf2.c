#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <libelf.h>
#include <gelf.h>

#include "cpu-common.h"
#include "log.h"
#include "cpu-elf2.h"
#include "cpu-utils.h"

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
    uint64_t compressed_size; // Compression related information
    uint16_t minor;
    uint16_t major;
    uint32_t arch;
    uint32_t obj_name_offset;
    uint32_t obj_name_len;
    uint64_t flags;
    uint64_t zero;      // Alignment for compression?
    uint64_t decompressed_len;  // Length of compressed data. There is an uncompressed footer
                              // so this is generally smaller than fatbin_size
};

#define FATBIN_FLAG_64BIT     0x0000000000000001LL
#define FATBIN_FLAG_DEBUG     0x0000000000000002LL
#define FATBIN_FLAG_LINUX     0x0000000000000010LL
#define FATBIN_FLAG_COMPRESS  0x0000000000002000LL

int elf2_init(void)
{
    if (elf_version(EV_CURRENT) == EV_NONE) {
        LOGE(LOG_ERROR, "ELF library initialization failed: %s", elf_errmsg(-1));
        return -1;
    }
}

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
    LOGE(LOG_DBG(1), "fat_text_header: fatbin_kind: %#x, header_size %#x, fatbin_size %#x, compressed_size %#x,\
        minor %#x, major %#x, arch %d, flags %#x, compressed_len %#x",
        fth->kind,
        fth->header_size,
        fth->fatbin_size,
        fth->compressed_size,
        fth->minor,
        fth->major,
        fth->arch,
        fth->flags,
        fth->decompressed_len);
    LOGE(LOG_DBG(1), "unknown fields: unknown1: %#x, zeros: %#x",
        fth->unknown1,
        fth->zero);
    fat_ptr += sizeof(struct fat_header);
    *fat_text_body_ptr = fat_text_header_ptr + fth->header_size;
    if (fth->flags & FATBIN_FLAG_DEBUG || fth->flags & FATBIN_FLAG_COMPRESS) {
       LOGE(LOG_DBG(1), "skipping extra byte \"%#02x\"", *((uint8_t*)*fat_text_body_ptr));
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

int elf2_get_fatbin_info(struct fat_header *fatbin, list *kernel_infos, void** fatbin_mem, unsigned* fatbin_size)
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

    if (fat_text_header->flags & FATBIN_FLAG_COMPRESS) {
        LOGE(LOG_WARNING, "fatbin contains compressed device code. This is not supported yet.");
        //return -1;
    }
    if (fat_text_header->flags & FATBIN_FLAG_DEBUG) {
        LOGE(LOG_WARNING, "fatbin contains debug information. This is not supported yet.");
        return -1;
    }

    if (elf2_parameter_info(kernel_infos, fat_text_body_ptr, fat_elf_header->fat_size) != 0) {
        LOGE(LOG_ERROR, "error getting parameter info");
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

static void print_hexmem(void *mem, size_t len)
{
    for (int i=0; i<len; i++) {
        printf("%02x ", ((uint8_t*)mem)[i]);
    }
    printf("\n");
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


static int get_section_by_name(Elf *elf, const char *name, Elf_Scn **section)
{
    Elf_Scn *scn = NULL;
    GElf_Shdr shdr;
    char *section_name = NULL;
    size_t str_section_index;

    if (elf == NULL || name == NULL || section == NULL) {
        LOGE(LOG_ERROR, "invalid argument");
        return -1;
    }

    if (elf_getshdrstrndx(elf, &str_section_index) != 0) {
        LOGE(LOG_ERROR, "elf_getshstrndx Wfailed");
        return -1;
    }

    while ((scn = elf_nextscn(elf, scn)) != NULL) {
        if (gelf_getshdr(scn, &shdr) != &shdr) {
            LOGE(LOG_ERROR, "gelf_getshdr failed");
            return -1;
        }
        if ((section_name = elf_strptr(elf, str_section_index, shdr.sh_name)) == NULL) {
            LOGE(LOG_ERROR, "elf_strptr failed");
            return -1;
        }
        //printf("%s, %#0x %#0x\n", section_name, shdr.sh_flags, shdr.sh_type);
        if (strcmp(section_name, name) == 0) {
            *section = scn;
            return 0;
        }
    }
    return -1;
}

static int get_parm_for_kernel(Elf *elf, kernel_info_t *kernel, void* memory, size_t memsize)
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
    int ret = -1;
    char *section_name = NULL;
    Elf_Scn *section = NULL;
    Elf_Data *data = NULL;

    if (kernel == NULL || kernel->name == NULL || memory == NULL) {
        LOGE(LOG_ERROR, "at least one parameter is NULL");
        goto cleanup;
    }
    kernel->param_num = 0;
    kernel->param_offsets = NULL;
    kernel->param_sizes = NULL;

    if (asprintf(&section_name, ".nv.info.%s", kernel->name) == -1) {
        LOGE(LOG_ERROR, "asprintf failed");
        goto cleanup;
    }

    if (get_section_by_name(elf, section_name, &section) != 0) {
        LOGE(LOG_ERROR, "section %s not found", section_name);
        goto cleanup;
    }

    if ((data = elf_getdata(section, NULL)) == NULL) {
        LOGE(LOG_ERROR, "error getting section data");
        goto cleanup;
    }

    //print_hexmem(data->d_buf, data->d_size);

    size_t secpos=0;
    int i=0;
    while (secpos < data->d_size) {
        struct nv_info_kernel_entry *entry = (struct nv_info_kernel_entry*)(data->d_buf+secpos);
        // printf("entry %d: format: %#x, attr: %#x, ", i++, entry->format, entry->attribute);
        if (entry->format == EIFMT_SVAL && entry->attribute == EIATTR_KPARAM_INFO) {
            if (entry->values_size != 0xc) {
                LOGE(LOG_ERROR, "EIATTR_KPARAM_INFO values size has not the expected value of 0xc");
                goto cleanup;
            }
            struct nv_info_kparam_info *kparam = (struct nv_info_kparam_info*)&entry->values;
            // printf("kparam: index: %#x, ordinal: %#x, offset: %#x, unknown: %#0x, cbank: %#0x, size: %#0x\n",
            //     kparam->index, kparam->ordinal, kparam->offset, kparam->unknown, kparam->cbank, kparam->size);
            LOGE(LOG_DBG(1), "param %d: offset: %#x, size: %#x", kparam->ordinal, kparam->offset, kparam->size);
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
    // printf("remaining: %d\n", data->d_size % sizeof(struct nv_info_kernel_entry));
    ret = 0;
 cleanup:
    free(section_name);
    return ret;
}


static int get_symtab(Elf *elf, Elf_Data **symbol_table_data, size_t *symbol_table_size, GElf_Shdr *symbol_table_shdr)
{
    GElf_Shdr shdr;
    Elf_Scn *section = NULL;

    if (elf == NULL || symbol_table_data == NULL || symbol_table_size == NULL) {
        LOGE(LOG_ERROR, "invalid argument");
        return -1;
    }

    if (get_section_by_name(elf, ".symtab", &section) != 0) {
        LOGE(LOG_ERROR, "could not find .nv.info section");
        return -1;
    }

    if (gelf_getshdr(section, &shdr) == NULL) {
        LOGE(LOG_ERROR, "gelf_getshdr failed");
        return -1;
    }

    if (symbol_table_shdr != NULL) {
        *symbol_table_shdr = shdr;
    }

    if(shdr.sh_type != SHT_SYMTAB) {
        LOGE(LOG_ERROR, "not a symbol table: %d", shdr.sh_type);
        return -1;
    }

    if ((*symbol_table_data = elf_getdata(section, NULL)) == NULL) {
        LOGE(LOG_ERROR, "elf_getdata failed");
        return -1;
    }

    *symbol_table_size = shdr.sh_size / shdr.sh_entsize;

    return 0;
}

static void print_symtab(Elf *elf)
{
    GElf_Sym sym;
    Elf_Data *symbol_table_data = NULL;
    GElf_Shdr shdr;
    size_t symnum;
    int i = 0;

    if (get_symtab(elf, &symbol_table_data, &symnum, &shdr) != 0) {
        LOGE(LOG_ERROR, "could not get symbol table");
        return;
    }

    LOGE(LOG_DEBUG, "found %d symbols", symnum);

    while (gelf_getsym(symbol_table_data, i, &sym) != NULL) {
        printf("sym %d: name: %s, value: %#x, size: %#x, info: %#x, other: %#x, shndx: %#x\n", i,
               elf_strptr(elf, shdr.sh_link, sym.st_name),
               sym.st_value, sym.st_size, sym.st_info, sym.st_other, sym.st_shndx);
        i++;
    }
}

static int check_elf(Elf *elf)
{
    Elf_Kind ek;
    GElf_Ehdr ehdr;

    int elfclass;
    char *id;
    size_t program_header_num;
    size_t sections_num;
    size_t section_str_num;
    int ret = -1;

    if ((ek = elf_kind(elf)) != ELF_K_ELF) {
        LOGE(LOG_ERROR, "elf_kind is not ELF_K_ELF, but %d", ek);
        goto cleanup;
    }

    if (gelf_getehdr(elf, &ehdr) == NULL) {
        LOGE(LOG_ERROR, "gelf_getehdr failed");
        goto cleanup;
    }

    if ((elfclass = gelf_getclass(elf)) == ELFCLASSNONE) {
        LOGE(LOG_ERROR, "gelf_getclass failed");
        goto cleanup;
    }

    if ((id = elf_getident(elf, NULL)) == NULL) {
        LOGE(LOG_ERROR, "elf_getident failed");
        goto cleanup;
    }

    LOGE(LOG_DBG(1), "elfclass: %d-bit; elf ident[0..%d]: %7s",
        (elfclass == ELFCLASS32) ? 32 : 64,
        EI_ABIVERSION, id);

    if (elf_getshdrnum(elf, &sections_num) != 0) {
        LOGE(LOG_ERROR, "elf_getphdrnum failed");
        goto cleanup;
    }

    if (elf_getphdrnum(elf, &program_header_num) != 0) {
        LOGE(LOG_ERROR, "elf_getshdrnum failed");
        goto cleanup;
    }

    if (elf_getshdrstrndx(elf, &section_str_num) != 0) {
        LOGE(LOG_ERROR, "elf_getshstrndx Wfailed");
        goto cleanup;
    }

    LOGE(LOG_DBG(1), "elf contains %d sections, %d program_headers, string table section: %d",
        sections_num, program_header_num, section_str_num);

    ret = 0;
cleanup:
    return ret;
}

int elf2_parameter_info(list *kernel_infos, void* memory, size_t memsize)
{
    struct __attribute__((__packed__)) nv_info_entry{
        uint8_t format;
        uint8_t attribute;
        uint16_t values_size;
        uint32_t kernel_id;
        uint32_t value;
    };

    Elf *elf = NULL;
    Elf_Scn *section = NULL;
    Elf_Data *data = NULL, *symbol_table_data = NULL;
    GElf_Shdr symtab_shdr;
    size_t symnum;
    int i = 0;
    GElf_Sym sym;

    int ret = -1;
    kernel_info_t *ki = NULL;
    const char *kernel_str;

    if (memory == NULL || memsize == 0) {
        LOGE(LOG_ERROR, "memory was NULL or memsize was 0");
        return -1;
    }

//#define ELF_DUMP_TO_FILE 1

#ifdef ELF_DUMP_TO_FILE
    FILE* fd2 = fopen("/tmp/cricket-elf-dump", "wb");
    fwrite(memory-1, memsize, 1, fd2);
    fclose(fd2);
#endif


    if ((elf = elf_memory(memory, memsize)) == NULL) {
        LOGE(LOG_ERROR, "elf_memory failed");
        goto cleanup;
    }

    if (check_elf(elf) != 0) {
        LOGE(LOG_ERROR, "check_elf failed");
        goto cleanup;
    }

    //print_symtab(elf);

    if (get_symtab(elf, &symbol_table_data, &symnum, &symtab_shdr) != 0) {
        LOGE(LOG_ERROR, "could not get symbol table");
        goto cleanup;
    }

    if (get_section_by_name(elf, ".nv.info", &section) != 0) {
        LOGE(LOG_ERROR, "could not find .nv.info section");
        goto cleanup;
    }

    if ((data = elf_getdata(section, NULL)) == NULL) {
        LOGE(LOG_ERROR, "elf_getdata failed");
        goto cleanup;
    }

    for (size_t secpos=0; secpos < data->d_size; secpos += sizeof(struct nv_info_entry)) {
        struct nv_info_entry *entry = (struct nv_info_entry *)(data->d_buf+secpos);
        LOGE(LOG_DBG(1), "%d: format: %#x, attr: %#x, values_size: %#x kernel: %#x, sval: %#x(%d)", 
        i++, entry->format, entry->attribute, entry->values_size, entry->kernel_id, 
        entry->value, entry->value);

        if (entry->values_size != 8) {
            LOGE(LOG_ERROR, "unexpected values_size: %#x", entry->values_size);
            continue;
        }

        if (entry->attribute != EIATTR_FRAME_SIZE) {
            continue;
        }

        if (entry->kernel_id >= symnum) {
            LOGE(LOG_ERROR, "kernel_id out of bounds: %#x", entry->kernel_id);
            continue;
        }

        if (gelf_getsym(symbol_table_data, entry->kernel_id, &sym) == NULL) {
            LOGE(LOG_ERROR, "gelf_getsym failed for entry %d", entry->kernel_id);
            continue;
        }
        if ((kernel_str = elf_strptr(elf, symtab_shdr.sh_link, sym.st_name) ) == NULL) {
            LOGE(LOG_ERROR, "strptr failed for entry %d", entry->kernel_id);
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

        if (get_parm_for_kernel(elf, ki, memory, memsize) != 0) {
            LOGE(LOG_ERROR, "get_parm_for_kernel failed for kernel %s", kernel_str);
            goto cleanup;
        }
    }

    ret = 0;
 cleanup:
    if (elf != NULL) {
        elf_end(elf);
    }
    return ret;
}