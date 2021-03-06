/* Copyright 2020 Sergei Bastrakov
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/warp/Traits.hpp>

#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

#include <catch2/catch.hpp>

#include <cstdint>

//#############################################################################
class ActivemaskSingleThreadWarpTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success) const
    -> void
    {
        std::int32_t const warpExtent = alpaka::warp::getSize(acc);
        ALPAKA_CHECK(*success, warpExtent == 1);

        ALPAKA_CHECK(*success, alpaka::warp::activemask(acc) == 1u);
    }
};

//#############################################################################
class ActivemaskMultipleThreadWarpTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success,
        std::uint64_t inactiveThreadIdx) const
    -> void
    {
        std::int32_t const warpExtent = alpaka::warp::getSize(acc);
        ALPAKA_CHECK(*success, warpExtent > 1);

        // Test relies on having a single warp per thread block
        auto const blockExtent = alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        ALPAKA_CHECK(*success, static_cast<std::int32_t>(blockExtent.prod()) == warpExtent);
        auto const localThreadIdx = alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const threadIdxInWarp = static_cast<std::uint64_t>(
            alpaka::idx::mapIdx<1u>(
                localThreadIdx,
                blockExtent)[0]
        );

        if (threadIdxInWarp == inactiveThreadIdx)
            return;

        auto const actual = alpaka::warp::activemask(acc);
        using Result = decltype(actual);
        Result const allActive =
            (Result{1} << static_cast<Result>(warpExtent)) - 1;
        Result const expected = allActive &
            ~(Result{1} << inactiveThreadIdx);
        ALPAKA_CHECK(
            *success,
            actual == expected);
    }
};

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "activemask", "[warp]", alpaka::test::acc::TestAccs)
{
    using Acc = TestType;
    using Dev = alpaka::dev::Dev<Acc>;
    using Pltf = alpaka::pltf::Pltf<Dev>;
    using Dim = alpaka::dim::Dim<Acc>;
    using Idx = alpaka::idx::Idx<Acc>;

    Dev const dev(alpaka::pltf::getDevByIdx<Pltf>(0u));
    auto const warpExtent = alpaka::dev::getWarpSize(dev);
    if (warpExtent == 1)
    {
        Idx const gridThreadExtentPerDim = 4;
        alpaka::test::KernelExecutionFixture<Acc> fixture(
            alpaka::vec::Vec<Dim, Idx>::all(gridThreadExtentPerDim));
        ActivemaskSingleThreadWarpTestKernel kernel;
        REQUIRE(
            fixture(
                kernel));
    }
    else
    {
        // Work around gcc 7.5 trying and failing to offload for OpenMP 4.0
#if BOOST_COMP_GNUC && (BOOST_COMP_GNUC == BOOST_VERSION_NUMBER(7, 5, 0)) && ALPAKA_ACC_CPU_BT_OMP4_ENABLED
        return;
#else
        using ExecutionFixture = alpaka::test::KernelExecutionFixture<Acc>;
        auto const gridBlockExtent = alpaka::vec::Vec<Dim, Idx>::all(2);
        // Enforce one warp per thread block
        auto blockThreadExtent = alpaka::vec::Vec<Dim, Idx>::ones();
        blockThreadExtent[0] = static_cast<Idx>(warpExtent);
        auto const threadElementExtent = alpaka::vec::Vec<Dim, Idx>::ones();
        auto workDiv = typename ExecutionFixture::WorkDiv{
            gridBlockExtent,
            blockThreadExtent,
            threadElementExtent};
        auto fixture = ExecutionFixture{ workDiv };
        ActivemaskMultipleThreadWarpTestKernel kernel;
        for (auto inactiveThreadIdx = 0u; inactiveThreadIdx < warpExtent;
            inactiveThreadIdx++)
            REQUIRE(
                fixture(
                    kernel,
                    inactiveThreadIdx));
#endif
    }
}
