#include <stdio.h>
#include <assert.h>
#include "list.h"
#include "log.h"

struct tuple{
    int a;
    double b;
};

void remove_tuple_elements(list *l)
{
    struct tuple *etuple;
    int cap = l->capacity;
    int ret;

    int initial_len = l->length;
    for (int i = 0; i < 4; i++) {
        ret = list_rm(l, 0);
        assert(ret == 0 && etuple != NULL && l->length == initial_len-i-1);
    }
    initial_len = l->length;
    for (int i = 0; i < l->length; i++) {
        ret = list_rm(l, i);
        assert(ret == 0 && etuple != NULL && l->length == initial_len-i-1);
    }
    assert(l->length == initial_len/2);
    for (int i = 0; i < l->length; ++i) {
        ret = list_at(l, i, (void**)&etuple);
        assert(ret == 0 && etuple != NULL);
        assert(list_get(l, i) == etuple);
        assert(etuple->a == i*2+1 && etuple->b == i*200.+100.);
    }

}

void add_tuple_elements(list *l)
{
    struct tuple *etuple;
    struct tuple enew;
    int cap = l->capacity;
    int ret;
    int length;

    for (int i = l->length; i < cap+1; ++i) {
        ret = list_append(l, (void**)&etuple);
        assert(ret == 0 && etuple != NULL && l->length == i+1);
        etuple->a = i;
        etuple->b = i*100.;
    }
    assert(l->capacity > cap);
    for (int i = 0; i < l->length; ++i) {
        ret = list_at(l, i, (void**)&etuple);
        assert(ret == 0 && etuple != NULL);
        assert(list_get(l, i) == etuple);
        assert(etuple->a == i && etuple->b == i*100.);
    }
    cap = l->capacity;
    for (int i = l->length; i < cap+1; ++i) {
        enew.a = i;
        enew.b = i*100.;
        ret = list_append_copy(l, &enew);
        assert(ret == 0 && l->length == i+1);
    }
    assert(l->capacity > cap);
    for (int i = 0; i < l->length; ++i) {
        ret = list_at(l, i, (void**)&etuple);
        assert(ret == 0 && etuple != NULL);
        assert(list_get(l, i) == etuple);
        assert(etuple->a == i && etuple->b == i*100.);
    }

    length = l->length;
    for (int i = 0; i < 4; ++i) {
        enew.a = i;
        enew.b = i*100.;
        ret = list_insert(l, i, &enew);
        assert(ret == 0 && l->length == length+i+1);
    }
    for (int i = 0; i < 4; ++i) {
        ret = list_at(l, i, (void**)&etuple);
        assert(ret == 0 && etuple != NULL);
        assert(list_get(l, i) == etuple);
        assert(etuple->a == i && etuple->b == i*100);
    }
}

void remove_int_elements(list *l)
{
    int *eint;
    int cap = l->capacity;
    int ret;

    int initial_len = l->length;
    for (int i = 0; i < 4; i++) {
        ret = list_rm(l, 0);
        assert(ret == 0 && eint != NULL && l->length == initial_len-i-1);
    }
    initial_len = l->length;
    for (int i = 0; i < l->length; i++) {
        ret = list_rm(l, i);
        assert(ret == 0 && eint != NULL && l->length == initial_len-i-1);
    }
    assert(l->length == initial_len/2);
    for (int i = 0; i < l->length; ++i) {
        ret = list_at(l, i, (void**)&eint);
        assert(ret == 0 && eint != NULL);
        assert(list_get(l, i) == eint);
        assert(*eint == i*2+1);
    }

}

void add_int_elements(list *l)
{
    int *eint;
    int cap = l->capacity;
    int length;
    int ret;

    for (int i = l->length; i < cap+1; ++i) {
        ret = list_append(l, (void**)&eint);
        assert(ret == 0 && eint != NULL && l->length == i+1);
        *eint = i;
    }
    assert(l->capacity > cap);
    for (int i = 0; i < l->length; ++i) {
        ret = list_at(l, i, (void**)&eint);
        assert(ret == 0 && eint != NULL);
        assert(list_get(l, i) == eint);
        assert(*eint == i);
    }
    cap = l->capacity;
    for (int i = l->length; i < cap+1; ++i) {
        ret = list_append_copy(l, &i);
        assert(ret == 0 && eint != NULL && l->length == i+1);
    }
    assert(l->capacity > cap);
    for (int i = 0; i < l->length; ++i) {
        ret = list_at(l, i, (void**)&eint);
        assert(ret == 0 && eint != NULL);
        assert(list_get(l, i) == eint);
        assert(*eint == i);
    }

    length = l->length;    
    for (int i = 0; i < 4; ++i) {
        ret = list_insert(l, i, &i);
        assert(ret == 0 && l->length == length+i+1);
    }
    for (int i = 0; i < 4; ++i) {
        ret = list_at(l, i, (void**)&eint);
        assert(ret == 0 && eint != NULL);
        assert(list_get(l, i) == eint);
        assert(*eint == i);
    }
}

int main()
{
    list l;
    int ret;
    ret = list_init(&l, sizeof(int));
    assert(ret == 0);
    assert(l.length == 0);
    add_int_elements(&l);
    remove_int_elements(&l);
    printf(".\n");
    list_free(&l);
    LOG(LOG_INFO, "list passed.");

    ret = list_init(&l, sizeof(struct tuple));
    assert(ret == 0);
    assert(l.length == 0);
    add_tuple_elements(&l);
    remove_tuple_elements(&l);
    list_free(&l);
    LOG(LOG_INFO, "list with complex datatype passed.");
    return 0;
}
