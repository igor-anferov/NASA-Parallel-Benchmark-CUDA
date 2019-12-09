#ifndef __TIMERS_H__
#define __TIMERS_H__

#ifdef __cplusplus
extern "C" {
#endif

void timer_clear( int n );
void timer_start( int n );
void timer_stop( int n );
double timer_read( int n );

#ifdef __cplusplus
}
#endif

#endif

