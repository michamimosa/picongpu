/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Bert Wesarg
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#include <alpaka/core/BoostPredef.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    #include <cuda_runtime.h>
    #if !BOOST_LANG_CUDA
        #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
    #endif
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)

    #if BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(9, 0, 0)
        #include <cuda_runtime_api.h>
    #else
        #if BOOST_COMP_HIP
            #include <hip/math_functions.h>
        #else
            #include <math_functions.hpp>
        #endif
    #endif
    
    #if !BOOST_LANG_HIP
        #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
    #endif
#endif

#include <alpaka/math/max/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA built in max.
        class MaxUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathMax, MaxUniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library integral max trait specialization.
            template<
                typename Tx,
                typename Ty>
            struct Max<
                MaxUniformCudaHipBuiltIn,
                Tx,
                Ty,
                std::enable_if_t<
                    std::is_integral<Tx>::value
                    && std::is_integral<Ty>::value>>
            {
                __device__ static auto max(
                    MaxUniformCudaHipBuiltIn const & max_ctx,
                    Tx const & x,
                    Ty const & y)
                -> decltype(::max(x, y))
                {
                    alpaka::ignore_unused(max_ctx);
                    return ::max(x, y);
                }
            };
            //#############################################################################
            //! The CUDA mixed integral floating point max trait specialization.
            template<
                typename Tx,
                typename Ty>
            struct Max<
                MaxUniformCudaHipBuiltIn,
                Tx,
                Ty,
                std::enable_if_t<
                    std::is_arithmetic<Tx>::value
                    && std::is_arithmetic<Ty>::value
                    && !(std::is_integral<Tx>::value
                        && std::is_integral<Ty>::value)>>
            {
                __device__ static auto max(
                    MaxUniformCudaHipBuiltIn const & max_ctx,
                    Tx const & x,
                    Ty const & y)
                -> decltype(::fmax(x, y))
                {
                    alpaka::ignore_unused(max_ctx);
                    return ::fmax(x, y);
                }
            };
        }
    }
}

#endif
