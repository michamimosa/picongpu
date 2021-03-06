/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/math/sqrt/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <type_traits>
#include <cmath>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library sqrt.
        class SqrtStdLib : public concepts::Implements<ConceptMathSqrt, SqrtStdLib>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library sqrt trait specialization.
            template<
                typename TArg>
            struct Sqrt<
                SqrtStdLib,
                TArg,
                std::enable_if_t<
                    std::is_arithmetic<TArg>::value>>
            {
                ALPAKA_FN_HOST static auto sqrt(
                    SqrtStdLib const & sqrt_ctx,
                    TArg const & arg)
                {
                    alpaka::ignore_unused(sqrt_ctx);
                    return std::sqrt(arg);
                }
            };
        }
    }
}
