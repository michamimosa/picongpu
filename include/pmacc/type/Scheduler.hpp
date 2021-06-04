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

// todo: cleanup includes
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

#include <fmt/format.h>

namespace pmacc
{
    enum SchedulingTags
    {
        SCHED_MPI,
        SCHED_CUPLA
    };

    using RedGrapesManager = redGrapes::Manager<
        redGrapes::LabelProperty,
        redGrapes::scheduler::SchedulingTagProperties<SchedulingTags>,
        redGrapes::helpers::cupla::CuplaTaskProperties>;

    using TaskProperties = typename RedGrapesManager::TaskProps;

} // namespace pmacc

template<>
struct fmt::formatter<pmacc::SchedulingTags>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(pmacc::SchedulingTags const& tag, FormatContext& ctx)
    {
        switch(tag)
        {
        case pmacc::SCHED_MPI:
            return fmt::format_to(ctx.out(), "\"MPI\"");
        case pmacc::SCHED_CUPLA:
            return fmt::format_to(ctx.out(), "\"Cupla\"");
        default:
            return fmt::format_to(ctx.out(), "\"undefined\"");
        }
    }
};
