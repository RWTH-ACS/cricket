#!/bin/bash

cp cpu_rpc_prot_svc.c cpu_rpc_prot_svc_mod.c
echo "void rpc_dispatch(struct svc_req *rqstp, xdrproc_t *ret_arg, xdrproc_t *ret_res, size_t *res_sz, bool_t (**ret_fun)(char *, void *, struct svc_req *))" >> cpu_rpc_prot_svc_mod.c
grep -Poz "(?s)rpc_cd_prog_1\(struct svc_req \*rqstp, register SVCXPRT \*transp\)\n\K.*?(?=svcerr_noproc \(transp\);)" cpu_rpc_prot_svc.c | tr -d '\000' >> cpu_rpc_prot_svc_mod.c
echo "_xdr_argument = NULL; \
    _xdr_result = NULL;
    local = NULL; \
    }
*ret_arg = _xdr_argument;\
*ret_res = _xdr_result;\
*res_sz = sizeof(result); \
*ret_fun = local;\
}" >> cpu_rpc_prot_svc_mod.c
