
#pragma once

#include <pmacc/types.hpp>

namespace pmacc
{
  namespace NEW{

struct Task
{
  virtual ~Task() {};
  virtual void properties( Scheduler::SchedulablePtr s ) {}
  virtual void run() {};
};

template <
    typename Task,
    typename... Args
>
auto enqueue_task( Args&&... args )
{
    Task task( std::forward( args )... );
    return Scheduler::enqueue_functor(
               [task]() { task.run(); },
               [task](Scheduler::SchedulablePtr s) { task.properties(s); }
           );
}

  }

} // namespace pmacc

