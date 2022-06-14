#ifndef _SCHED_H_
#define _SCHED_H_

typedef struct _sched_t {
    int (*init)(void);
    int (*retain)(int id);
    int (*release)(int id);
    int (*rm)(int id);
    void (*deinit)(void);
} sched_t;

sched_t *sched;

#define SCHED_RETAIN sched->retain(rqstp->rq_xprt->xp_fd)
#define SCHED_RELEASE sched->release(rqstp->rq_xprt->xp_fd)

#endif //_SCHED_H_
