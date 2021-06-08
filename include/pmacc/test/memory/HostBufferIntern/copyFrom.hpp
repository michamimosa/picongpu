/* Copyright 2015-2021 Erik Zenker
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

/* #includes in "test/memoryUT.cu" */

namespace pmacc
{
    namespace test
    {
        namespace memory
        {
            namespace HostBufferIntern
            {
                /**
                 * Checks if data is copied correctly from device to
                 * host.
                 */
                struct CopyFromTest
                {
                    template<typename T_Dim>
                    void exec(T_Dim)
                    {
                        using Data = uint8_t;
                        using Extents = size_t;

                        using ::pmacc::test::memory::getElementsPerDim;

                        std::vector<size_t> nElementsPerDim = getElementsPerDim<T_Dim>();

                        for(unsigned i = 0; i < nElementsPerDim.size(); ++i)
                        {
                            ::pmacc::DataSpace<T_Dim::value> const dataSpace
                                = ::pmacc::DataSpace<T_Dim::value>::create(nElementsPerDim[i]);

                            ::pmacc::memory::HostBuffer<Data, T_Dim::value> hostBuffer(dataSpace);
                            ::pmacc::memory::DeviceBuffer<Data, T_Dim::value> deviceBuffer(dataSpace);

                            ::pmacc::Environment<>::task(
                                [dataSpace](auto buf) {
                                    for(size_t i = 0; i < static_cast<size_t>(dataSpace.productOfComponents()); ++i)
                                    {
                                        hostBufferIntern->getPointer()[i] = static_cast<Data>(i);
                                    }
                                },
                                hostBuffer.write());

                            ::pmacc::memory::buffers::copy( deviceBuffer.write(), hostBuffer.read() );
                            ::pmacc::memory::buffers::reset( hostBuffer.write() );
                            ::pmacc::memory::buffers::copy( hostBuffer.write(), deviceBuffer.read() );

                            ::pmacc::Environment<>::task(
                                    [dataSpace](auto bufData) {
                                        for(size_t i = 0; i < static_cast<size_t>(dataSpace.productOfComponents());
                                            ++i)
                                        {
                                            REQUIRE(bufData->getPointer()[i] == static_cast<Data>(i));
                                        }
                                    },
                                    hostBuffer.data().read());
                        }
                    }

                    PMACC_NO_NVCC_HDWARNING
                    template<typename T_Dim>
                    HDINLINE void operator()(T_Dim dim)
                    {
                        exec(dim);
                    }
                };

            } // namespace HostBufferIntern
        } // namespace memory
    } // namespace test
} // namespace pmacc

TEST_CASE("HostBufferIntern::copyFrom", "[copyFrom]")
{
    using namespace pmacc::test::memory::HostBufferIntern;
    ::boost::mpl::for_each<Dims>(CopyFromTest());
}
