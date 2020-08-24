/* Copyright 2019-2020 Michael Sippel
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <redGrapes/resource/ioresource.hpp>
#include <redGrapes/property/inherit.hpp>
#include <redGrapes/property/resource.hpp>
#include <redGrapes/property/label.hpp>
#include <redGrapes/graph/scheduling_graph.hpp>

#include <redGrapes/scheduler/default_scheduler.hpp>
#include <redGrapes/scheduler/tag_match.hpp>
#include <redGrapes/helpers/cupla/scheduler.hpp>
#include <redGrapes/helpers/mpi/scheduler.hpp>
#include <redGrapes/manager.hpp>

namespace pmacc
{

enum SchedulingTags
{
    SCHED_MPI,
    SCHED_CUPLA
};

using TaskProperties = redGrapes::TaskProperties<
    redGrapes::ResourceProperty,
    redGrapes::LabelProperty,
    redGrapes::scheduler::SchedulingTagProperties<>,
    redGrapes::helpers::cupla::CuplaTaskProperties
>;

std::ostream& functor_backtrace(std::ostream& out);

struct EnqueuePolicy
{
    static bool is_serial(TaskProperties const & a, TaskProperties const & b)
    {
        return redGrapes::ResourceUser::is_serial( a, b );
    }

    static void assert_superset(TaskProperties const & super, TaskProperties const & sub)
    {
        if(! redGrapes::ResourceUser::is_superset( super, sub ))
        {
            std::stringstream stream;
            stream << "Not allowed: " << super.label << "is no superset of " << sub.label << std::endl
                   << super.label << " has access: " << std::endl
                   << (redGrapes::ResourceUser)super << std::endl << std::endl
                   << sub.label << " has access: " << std::endl
	           << (redGrapes::ResourceUser)sub << std::endl;
            throw std::runtime_error(stream.str());
        }
    }
};
    
} // namespace pmacc

