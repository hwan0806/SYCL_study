/*
    Author: Thomas Kim 김창희
    First Edit: Jan. 25, 2022
*/

#ifndef _CPG_OPENCL_SYCL_HPP
#define _CPG_OPENCL_SYCL_HPP

#ifndef NOMINMAX
#define NOMINMAX
#endif 

#include "cpg_types.hpp"

#ifndef __SYCL_INTERNAL_API
    #define __SYCL_INTERNAL_API
#endif

#include <cl/sycl.hpp>
#include <tbb/tbb.h>
#include <regex>

namespace cpg::opencl_sycl
{
    namespace cgt = cpg::types;

    using device_info = sycl::info::device;
    using access = sycl::access::mode;

    namespace hidden
    {
        template<typename = int>
        std::string file_contents(const char* fileName)
        {
            std::ifstream kernelFile(fileName, std::ios::in);

            if (!kernelFile.is_open())
            {
                std::cout << "Failed to open file for reading: " 
                    << fileName << std::endl;

                return "";
            }

            std::ostringstream oss;
            oss << kernelFile.rdbuf();
            std::string srcStdStr = oss.str();
            
            if(srcStdStr.back() != '\n') srcStdStr.push_back('\n');

            if(srcStdStr[0] == (char)0xEF && srcStdStr[1] == (char)0xBB && srcStdStr[2] == (char)0xBF)
                return std::string( &srcStdStr[3] );
            else
                return srcStdStr;
        }

        template<typename StringType>
        auto count_lines(StringType& src)
        {
            std::size_t lines = 0;

            for(auto c: src) if(c == '\n') ++lines;

            return lines;
        }

         template<typename FileNameType, typename CountType, std::size_t N,
            typename StringType>
        auto error_line_file_name(std::array<FileNameType, N> const& 
            file_names, std::array<CountType, N> const& line_counts, StringType error_line_str)
        {
            std::size_t error_line = std::stoll(error_line_str);
            std::size_t error_file_index = 0;

            for(std::size_t i = 1; i < N; ++i)
            {
                if(line_counts[i-1] < error_line && 
                    error_line <= line_counts[i])
                {
                    error_line -= line_counts[i-1];
                    error_file_index = i; break;
                }
            }

            std::ostringstream oss;
            oss << error_line;

            return std::tuple{ oss.str(), file_names[error_file_index] };
        }

        template<typename CharType>
        std::basic_string<CharType> replace_string(std::basic_string<CharType> target,
            std::basic_string<CharType> const& str_find, std::basic_string<CharType> const& str_replace)
        {
            auto pos = target.find(str_find);
          
            if(pos != std::basic_string<CharType>::npos)
                return target.replace(pos, str_find.size(), str_replace); 
            else
                return target;
        }

        template<typename FileNameType, typename CountType, std::size_t N,
            typename StringType>
        auto parse_error_message(std::array<FileNameType, N> const& 
            file_names, std::array<CountType, N> const& line_counts, StringType error_msg)
        {
            // 1:11:5: error:, 1:20:5: warning:
            std::regex pattern{ R"((\d+):(\d+):(\d+):)"  };

            auto result = error_msg;

            for(std::sregex_iterator p{error_msg.begin(), error_msg.end(), pattern};
                p != std::sregex_iterator{}; ++p)
                {
                    auto& m = *p;

                    auto [line, filename] = error_line_file_name(file_names, line_counts, m[2].str());

                    std::ostringstream oss;
                    oss<<"[" << filename <<"] "
                        << line <<":" 
                        << m[3].str()<<":"; 

                    result = replace_string(result, m.str(), oss.str());
                }          

            return result;
        }

        template<typename... OpenCLFiles>
        cl_program CreateProgram(cl_context context, cl_device_id device, OpenCLFiles... opencl_files)
        {
            cl_int errNum;
            cl_program program;
            
            constexpr std::size_t FileCount = sizeof...(opencl_files);

            std::array src_files{ opencl_files... };

            std::array std_file_contents { file_contents(opencl_files) ... };

            auto srcStr = cgt::for_stallion<FileCount>(
                    [&std_file_contents]<auto... i>( cgt::sequence<i...>)
            {
                return std::array{ std_file_contents[i].c_str()... };
            } );
            
            program = 
                clCreateProgramWithSource(context, FileCount, srcStr.data(), NULL, NULL);

            if(program == NULL)
            {
                std::cout << "Failed to create CL program from source." << std::endl;
                return NULL;
            }

            errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

            if (errNum != CL_SUCCESS)
            {
                std::size_t N;

                auto line_counts =
                    cgt::for_stallion<FileCount>([&std_file_contents]<auto... i>(cgt::sequence<i...>)
                    {
                        return std::array{ count_lines(std_file_contents[i])... };
                    });

                for(std::size_t i = 1; i < FileCount; ++i)
                    line_counts[i] += line_counts[i-1];

                // https://www.khronos.org/registry/OpenCL//sdk/2.2/docs/man/html/clGetProgramBuildInfo.html
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &N);

                std::string buildLog(N, ' ');

                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, N, buildLog.data(), NULL);
                
                buildLog = parse_error_message(src_files, line_counts, buildLog);
                
                std::cout << buildLog;

                clReleaseProgram(program);

                return NULL;
            }
            else
                return program;
        }

    }
    // end of namespace hidden

    template<typename SycleQueue, typename... OpenCLFiles>
    auto create_program(SycleQueue& queue, OpenCLFiles... opencl_files)
    {
        auto context = queue.get_context();
        auto device = queue.get_device();

        auto program = cgt::raii_create_object(
                hidden::CreateProgram(context.get(), device.get(),
                    opencl_files...), clReleaseProgram);
        
        if(program)
            clBuildProgram(program.get(), 0, nullptr, nullptr, nullptr, nullptr);
        
        return program;
    }

    template<typename ProgramType>
    auto create_kernel(ProgramType& program, const char* kernel_name)
    {
        return cgt::raii_create_object(
                clCreateKernel(program.get(), kernel_name, nullptr), clReleaseKernel);
    }

    auto sycl_kernel(auto& opencl_kernel, auto& queue)
    {
        return sycl::kernel{ opencl_kernel.get(), queue.get_context() }; 
    }
}
// end of namespace cpg::sycl
#endif // end of file _CPG_OPENCL_SYCL_HPP