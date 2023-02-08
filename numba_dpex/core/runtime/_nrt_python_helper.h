// SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Definition of NRT functions for marshalling from / to Python objects.
 * This module is included by _nrt_pythonmod.c and by pycc-compiled modules.
 */

#include "numba/_pymodule.h"
#include <stdatomic.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayscalars.h>
#include <numpy/ndarrayobject.h>

#include <numba/_arraystruct.h>
#include <numba/_numba_common.h>
#include <numba/core/runtime/nrt.h>

#define SYCL_USM_ARRAY_INTERFACE "__sycl_usm_array_interface__"

/*
 * Global resources.
 */
struct NRT_MemSys
{
    /* Shutdown flag */
    int shutting;
    /* Stats */
    struct
    {
        bool enabled;
        atomic_size_t alloc;
        atomic_size_t free;
        atomic_size_t mi_alloc;
        atomic_size_t mi_free;
    } stats;
    /* System allocation functions */
    struct
    {
        NRT_malloc_func malloc;
        NRT_realloc_func realloc;
        NRT_free_func free;
    } allocator;
};

/* The Memory System object */
struct NRT_MemSys TheMSys;

/*
 * The MemInfo structure.
 * NOTE: copy from numba/core/runtime/nrt.c
 */
struct MemInfo
{
    size_t refct;
    NRT_dtor_function dtor;
    void *dtor_info;
    void *data;
    size_t size; /* only used for NRT allocated memory */
    NRT_ExternalAllocator *external_allocator;
};

/*!
 * @brief A wrapper struct to store a MemInfo pointer along with the PyObject
 * that is associated with the MeMinfo.
 *
 * The struct is stored in the dtor_info attribute of a MemInfo object and
 * used by the destructor to free the MemInfo and DecRef the Pyobject.
 *
 */
typedef struct
{
    PyObject *owner;
    NRT_MemInfo *mi;
} MemInfoDtorInfo;

typedef struct
{
    PyObject_HEAD NRT_MemInfo *meminfo;
} MemInfoObject;

// following funcs are copied from numba/core/runtime/nrt.cpp
void *NRT_MemInfo_external_allocator(NRT_MemInfo *mi)
{
    NRT_Debug(nrt_debug_print(
        "NRT_MemInfo_external_allocator meminfo: %p external_allocator: %p\n",
        mi, mi->external_allocator));
    return mi->external_allocator;
}

void *NRT_MemInfo_data(NRT_MemInfo *mi) { return mi->data; }

void NRT_MemInfo_release(NRT_MemInfo *mi)
{
    assert(mi->refct > 0 && "RefCt cannot be 0");
    /* RefCt drop to zero */
    if ((--(mi->refct)) == 0) {
        NRT_MemInfo_call_dtor(mi);
    }
}

void NRT_MemInfo_call_dtor(NRT_MemInfo *mi)
{
    NRT_Debug(nrt_debug_print("NRT_MemInfo_call_dtor %p\n", mi));
    if (mi->dtor && !TheMSys.shutting)
        /* We have a destructor and the system is not shutting down */
        mi->dtor(mi->data, mi->size, mi->dtor_info);
    /* Clear and release MemInfo */
    NRT_MemInfo_destroy(mi);
}

void NRT_MemInfo_acquire(NRT_MemInfo *mi)
{
    // NRT_Debug(nrt_debug_print("NRT_MemInfo_acquire %p refct=%zu\n", mi,
    // mi->refct.load()));
    assert(mi->refct > 0 && "RefCt cannot be zero");
    mi->refct++;
}

size_t NRT_MemInfo_size(NRT_MemInfo *mi) { return mi->size; }

void *NRT_MemInfo_parent(NRT_MemInfo *mi) { return mi->dtor_info; }

size_t NRT_MemInfo_refcount(NRT_MemInfo *mi)
{
    /* Should never returns 0 for a valid MemInfo */
    if (mi && mi->data)
        return mi->refct;
    else {
        return (size_t)-1;
    }
}

void NRT_Free(void *ptr)
{
    NRT_Debug(nrt_debug_print("NRT_Free %p\n", ptr));
    TheMSys.allocator.free(ptr);
    if (TheMSys.stats.enabled) {
        TheMSys.stats.free++;
    }
}

void NRT_dealloc(NRT_MemInfo *mi)
{
    NRT_Debug(
        nrt_debug_print("NRT_dealloc meminfo: %p external_allocator: %p\n", mi,
                        mi->external_allocator));
    if (mi->external_allocator) {
        mi->external_allocator->free(mi, mi->external_allocator->opaque_data);
        if (TheMSys.stats.enabled) {
            TheMSys.stats.free++;
        }
    }
    else {
        NRT_Free(mi);
    }
}

void NRT_MemInfo_destroy(NRT_MemInfo *mi)
{
    NRT_dealloc(mi);
    if (TheMSys.stats.enabled) {
        TheMSys.stats.mi_free++;
    }
}

// following funcs are copied from numba/core/runtime/_nrt_python.c
static void MemInfo_dealloc(MemInfoObject *self)
{
    NRT_MemInfo_release(self->meminfo);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int MemInfo_init(MemInfoObject *self, PyObject *args, PyObject *kwds)
{
    static char *keywords[] = {"ptr", NULL};
    PyObject *raw_ptr_obj;
    void *raw_ptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", keywords, &raw_ptr_obj)) {
        return -1;
    }
    raw_ptr = PyLong_AsVoidPtr(raw_ptr_obj);
    NRT_Debug(
        nrt_debug_print("MemInfo_init self=%p raw_ptr=%p\n", self, raw_ptr));

    if (PyErr_Occurred())
        return -1;
    self->meminfo = (NRT_MemInfo *)raw_ptr;
    assert(NRT_MemInfo_refcount(self->meminfo) > 0 && "0 refcount");
    return 0;
}

static int MemInfo_getbuffer(PyObject *exporter, Py_buffer *view, int flags)
{
    Py_ssize_t len;
    void *buf;
    int readonly = 0;

    MemInfoObject *miobj = (MemInfoObject *)exporter;
    NRT_MemInfo *mi = miobj->meminfo;

    buf = NRT_MemInfo_data(mi);
    len = NRT_MemInfo_size(mi);
    return PyBuffer_FillInfo(view, exporter, buf, len, readonly, flags);
}

static PyBufferProcs MemInfo_bufferProcs = {MemInfo_getbuffer, NULL};

static PyObject *MemInfo_acquire(MemInfoObject *self)
{
    NRT_MemInfo_acquire(self->meminfo);
    Py_RETURN_NONE;
}

static PyObject *MemInfo_release(MemInfoObject *self)
{
    NRT_MemInfo_release(self->meminfo);
    Py_RETURN_NONE;
}

static PyObject *MemInfo_get_data(MemInfoObject *self, void *closure)
{
    return PyLong_FromVoidPtr(NRT_MemInfo_data(self->meminfo));
}

static PyObject *MemInfo_get_refcount(MemInfoObject *self, void *closure)
{
    size_t refct = NRT_MemInfo_refcount(self->meminfo);
    if (refct == (size_t)-1) {
        PyErr_SetString(PyExc_ValueError, "invalid MemInfo");
        return NULL;
    }
    return PyLong_FromSize_t(refct);
}

static PyObject *MemInfo_get_external_allocator(MemInfoObject *self,
                                                void *closure)
{
    void *p = NRT_MemInfo_external_allocator(self->meminfo);
    return PyLong_FromVoidPtr(p);
}

static PyObject *MemInfo_get_parent(MemInfoObject *self, void *closure)
{
    void *p = NRT_MemInfo_parent(self->meminfo);
    if (p) {
        Py_INCREF(p);
        return (PyObject *)p;
    }
    else {
        Py_INCREF(Py_None);
        return Py_None;
    }
}

static PyMethodDef MemInfo_methods[] = {
    {"acquire", (PyCFunction)MemInfo_acquire, METH_NOARGS,
     "Increment the reference count"},
    {"release", (PyCFunction)MemInfo_release, METH_NOARGS,
     "Decrement the reference count"},
    {NULL} /* Sentinel */
};

static PyGetSetDef MemInfo_getsets[] = {
    {"data", (getter)MemInfo_get_data, NULL,
     "Get the data pointer as an integer", NULL},
    {"refcount", (getter)MemInfo_get_refcount, NULL, "Get the refcount", NULL},
    {"external_allocator", (getter)MemInfo_get_external_allocator, NULL,
     "Get the external allocator", NULL},
    {"parent", (getter)MemInfo_get_parent, NULL, NULL},
    {NULL} /* Sentinel */
};

static PyTypeObject MemInfoType = {
    PyVarObject_HEAD_INIT(NULL, 0) "_dpexrt_python._MemInfo", /* tp_name */
    sizeof(MemInfoObject),                                    /* tp_basicsize */
    0,                                                        /* tp_itemsize */
    (destructor)MemInfo_dealloc,                              /* tp_dealloc */
    0,                                        /* tp_vectorcall_offset */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_as_async */
    0,                                        /* tp_repr */
    0,                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash */
    0,                                        /* tp_call */
    0,                                        /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    &MemInfo_bufferProcs,                     /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    0,                                        /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    MemInfo_methods,                          /* tp_methods */
    0,                                        /* tp_members */
    MemInfo_getsets,                          /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)MemInfo_init,                   /* tp_init */
    0,                                        /* tp_alloc */
    0,                                        /* tp_new */
    0,                                        /* tp_free */
    0,                                        /* tp_is_gc */
    0,                                        /* tp_bases */
    0,                                        /* tp_mro */
    0,                                        /* tp_cache */
    0,                                        /* tp_subclasses */
    0,                                        /* tp_weaklist */
    0,                                        /* tp_del */
    0,                                        /* tp_version_tag */
    0,                                        /* tp_finalize */
    /* The docs suggest Python 3.8 has no tp_vectorcall
     * https://github.com/python/cpython/blob/d917cfe4051d45b2b755c726c096ecfcc4869ceb/Doc/c-api/typeobj.rst?plain=1#L146
     * but the header has it:
     * https://github.com/python/cpython/blob/d917cfe4051d45b2b755c726c096ecfcc4869ceb/Include/cpython/object.h#L257
     */
    0, /* tp_vectorcall */
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 8)
    /* This is Python 3.8 only.
     * See: https://github.com/python/cpython/blob/3.8/Include/cpython/object.h
     * there's a tp_print preserved for backwards compatibility. xref:
     * https://github.com/python/cpython/blob/d917cfe4051d45b2b755c726c096ecfcc4869ceb/Include/cpython/object.h#L260
     */
    0, /* tp_print */
#endif

/* WARNING: Do not remove this, only modify it! It is a version guard to
 * act as a reminder to update this struct on Python version update! */
#if (PY_MAJOR_VERSION == 3)
#if !((PY_MINOR_VERSION == 8) || (PY_MINOR_VERSION == 9) ||                    \
      (PY_MINOR_VERSION == 10))
#error "Python minor version is not supported."
#endif
#else
#error "Python major version is not supported."
#endif
    /* END WARNING*/
};
