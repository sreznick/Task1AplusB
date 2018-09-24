#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cassert>


template <typename T>
std::string to_string(T value)
{
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line)
{
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


int main()
{    
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // TODO 1 По аналогии с заданием Example0EnumDevices узнайте какие есть устройства, и выберите из них какое-нибудь
    // (если есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)

    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    cl_device_id cpuDevice = 0;
    cl_device_id gpuDevice = 0;

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        cl_platform_id platform = 0;

        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        platform = platforms[platformIndex];

        size_t platformNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));

        std::vector<unsigned char> platformName(platformNameSize, 0);
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::cout << "Number of devices: " << devicesCount << std::endl;

        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount && (cpuDevice == 0 || gpuDevice == 0); ++deviceIndex) {
            std::cout << "Device #: " << deviceIndex << std::endl;

            cl_device_id device = devices[deviceIndex];

            size_t deviceNameSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));

            std::vector<unsigned char> deviceName(deviceNameSize, 0);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));

            std::cout << "    Device name: " << deviceName.data() << std::endl;

            cl_device_type deviceType;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));

            std::cout << "    Device type: " << deviceType << std::endl;

            if (deviceType & CL_DEVICE_TYPE_CPU && cpuDevice == 0) {
                cpuDevice = device;
            }

            if (deviceType & CL_DEVICE_TYPE_GPU && gpuDevice == 0) {
                gpuDevice = device;
            }
        }
    }


    std::cout << std::endl;
    std::cout << "CPU device: " << cpuDevice << std::endl;
    std::cout << "GPU device: " << gpuDevice << std::endl;

    cl_device_id device = gpuDevice ? gpuDevice : cpuDevice;

    std::cout << "Selected device: " << device << std::endl;

    if (!device) {
        std::cerr << "No devices was found" << std::endl;
        return 1;
    }


    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)

    const cl_context_properties properties[] = {0};
    cl_device_id devices[] = {device};
    cl_int errcode = 0;

    cl_context ctx = clCreateContext(properties, 1, devices, nullptr, nullptr, &errcode);

    std::cerr << "Context: " << ctx << std::endl;

    if (errcode != CL_SUCCESS) {
        std::cerr << "error creating context: " << errcode << std::endl;

        return 2;
    }


    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue

    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0 /* in-order */ , &errcode);

    std::cout << "Queue: " << queue << std::endl;
    if (errcode != CL_SUCCESS) {
        std::cerr << "error creating queue: " << errcode << std::endl;

        clReleaseContext(ctx);
        return 3;
    }

    unsigned int n = 100 * 1000*1000;
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    // TODO 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт что чисел в каждом массиве - n штук
    // Данные в as и bs можно прогрузить этим же методом скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    // И хорошо бы сразу добавить в конце clReleaseMemObject (аналогично все дальнейшие ресурсы вроде OpenCL под-программы, кернела и т.п. тоже нужно освобождать)

    cl_mem bufferA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, n * sizeof(float), as.data(), &errcode);

    std::cout << "Buffer A: " << bufferA << std::endl;
    if (errcode != CL_SUCCESS) {
        std::cerr << "error creating memory buffer A: " << errcode << std::endl;

        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);

        return 4;
    }

    cl_mem bufferB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, n * sizeof(float), bs.data(), &errcode);

    std::cout << "Buffer B: " << bufferB << std::endl;
    if (errcode != CL_SUCCESS) {
        std::cerr << "error creating memory buffer C: " << errcode << std::endl;

        clReleaseMemObject(bufferA);

        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);

        return 5;
    }

    cl_mem bufferC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n * sizeof(float), 0, &errcode);

    std::cout << "Buffer C: " << bufferC << std::endl;
    if (errcode != CL_SUCCESS) {
        std::cerr << "error creating memory buffer C: " << errcode << std::endl;

        clReleaseMemObject(bufferA);
        clReleaseMemObject(bufferB);

        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);


        return 6;
    }


    // TODO 6 Выполните TODO 5 (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания)
    // напечатав исходники в консоль (if проверяет что удалось считать хоть что-то)
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        std::cout << kernel_sources << std::endl;
    }

    // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание что передать вам нужно указатель на указатель
    const char *src = kernel_sources.c_str();
    cl_program program = clCreateProgramWithSource(ctx, 1, &src, nullptr, &errcode);
    std::cout << "Program: " << program << std::endl;
    if (errcode != CL_SUCCESS) {
        std::cerr << "error creating program: " << errcode << std::endl;

        clReleaseMemObject(bufferA);
        clReleaseMemObject(bufferB);
        clReleaseMemObject(bufferC);

        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);

        return 7;
    }

    
    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram
    cl_int build_status = clBuildProgram (program, 1, devices, nullptr, nullptr, nullptr);
    std::cout << "build_status: " << build_status << " " << std::endl;

    // А так же напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // см. clGetProgramBuildInfo
     size_t log_size = 0;
     cl_int buildInfoStatus = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

     if (buildInfoStatus == CL_SUCCESS) {

         std::cout << "Size of log: " << log_size << std::endl;

         std::vector<char> log(log_size, 0);

         buildInfoStatus = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);

         if (buildInfoStatus == CL_SUCCESS) {
             if (log_size > 1) {
                 std::cout << "Log:" << std::endl;
                 std::cout << log.data() << std::endl;
             }
         } else {
             std::cerr << "unable to determine log contents: " << buildInfoStatus << std::endl;
         }
     } else {
        std::cerr << "unable to determine log size: " << buildInfoStatus << std::endl;
     }

     std::cout << "------------------" << std::endl;


     if (build_status != CL_SUCCESS) {
        std::cerr << "error building program: " << build_status << std::endl;

        clReleaseProgram(program);

        clReleaseMemObject(bufferA);
        clReleaseMemObject(bufferB);
        clReleaseMemObject(bufferC);

        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);

        return 8;
     }

    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects
    cl_kernel kernels[] = {0};
    cl_uint n_kernels;

    cl_int kres = clCreateKernelsInProgram (program, 1, kernels, &n_kernels);

    std::cout << "# of kernels: " << n_kernels << std::endl;
    std::cout << "kernel: " << kernels[0] << std::endl;

    if (kres != CL_SUCCESS) {
       std::cerr << "error creating kernels: " << kres << std::endl;

       clReleaseProgram(program);

       clReleaseMemObject(bufferA);
       clReleaseMemObject(bufferB);
       clReleaseMemObject(bufferC);

       clReleaseCommandQueue(queue);
       clReleaseContext(ctx);

       return 9;
    }


    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь что тип количества элементов такой же в кернеле)
    {
        unsigned int i = 0;
        cl_int status;
        status = clSetKernelArg(kernels[0], i++, sizeof(cl_mem), &bufferA);

        if (status != CL_SUCCESS) {
            std::cerr << "error setting argument #" << (i - 1) << ": " << status << std::endl;

            for (int i=0; i<n_kernels; i++) {
                clReleaseKernel(kernels[i]);
            }


            clReleaseProgram(program);

            clReleaseMemObject(bufferA);
            clReleaseMemObject(bufferB);
            clReleaseMemObject(bufferC);

            clReleaseCommandQueue(queue);
            clReleaseContext(ctx);

            return 10;
        }

        status = clSetKernelArg(kernels[0], i++, sizeof(cl_mem), &bufferB);

        if (status != CL_SUCCESS) {
            std::cerr << "error setting argument #" << (i - 1) << ": " << status << std::endl;

            for (int i=0; i<n_kernels; i++) {
                clReleaseKernel(kernels[i]);
            }


            clReleaseProgram(program);

            clReleaseMemObject(bufferA);
            clReleaseMemObject(bufferB);
            clReleaseMemObject(bufferC);

            clReleaseCommandQueue(queue);
            clReleaseContext(ctx);

            return 11;
        }


        status = clSetKernelArg(kernels[0], i++, sizeof(cl_mem), &bufferC);
        std::cout << "status: " << status << std::endl;

        if (status != CL_SUCCESS) {
            std::cerr << "error setting argument #" << (i - 1) << ": " << status << std::endl;

            for (int i=0; i<n_kernels; i++) {
                clReleaseKernel(kernels[i]);
            }


            clReleaseProgram(program);

            clReleaseMemObject(bufferA);
            clReleaseMemObject(bufferB);
            clReleaseMemObject(bufferC);

            clReleaseCommandQueue(queue);
            clReleaseContext(ctx);

            return 12;
        }


        status = clSetKernelArg(kernels[0], i++, sizeof(unsigned int), &n);
        std::cout << "status: " << status << std::endl;

        if (status != CL_SUCCESS) {
            std::cerr << "error setting argument #" << (i - 1) << ": " << status << std::endl;

            for (int i=0; i<n_kernels; i++) {
                clReleaseKernel(kernels[i]);
            }


            clReleaseProgram(program);

            clReleaseMemObject(bufferA);
            clReleaseMemObject(bufferB);
            clReleaseMemObject(bufferC);

            clReleaseCommandQueue(queue);
            clReleaseContext(ctx);

            return 13;
        }

    }

    // TODO 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)
    
    // TODO 12 Запустите выполнения кернела:
    // - С одномерной рабочей группой размера 128
    // - В одномерном рабочем пространстве размера roundedUpN, где roundedUpN - наименьшее число кратное 128 и при этом не меньшее n
    // - см. clEnqueueNDRangeKernel
    // - Обратите внимание что чтобы дождаться окончания вычислений (чтобы знать когда можно смотреть результаты в cs_gpu) нужно:
    //   - Сохранить событие "кернел запущен" (см. аргумент "cl_event *event")
    //   - Дождаться завершения полунного события - см. в документации подходящий метод среди Event Objects
    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        size_t wgSizes[] = {workGroupSize};
        size_t workSizes[] = {global_work_size};

        timer t; // Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;

            cl_int status = clEnqueueNDRangeKernel(queue, kernels[0], 1, nullptr, workSizes, wgSizes, 0, nullptr, &event);

            if (status != CL_SUCCESS) {
                std::cerr << "error enqueing kernel at iteration #" << i << ": " << status << std::endl;

                for (int i=0; i<n_kernels; i++) {
                    clReleaseKernel(kernels[i]);
                }


                clReleaseProgram(program);

                clReleaseMemObject(bufferA);
                clReleaseMemObject(bufferB);
                clReleaseMemObject(bufferC);

                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);

                return 14;
            }


            cl_int estatus = clWaitForEvents(1, &event);

            if (estatus != CL_SUCCESS) {
                std::cerr << "error waiting for event at iteration #" << i << ": " << estatus << std::endl;

                for (int i=0; i<n_kernels; i++) {
                    clReleaseKernel(kernels[i]);
                }


                clReleaseProgram(program);

                clReleaseMemObject(bufferA);
                clReleaseMemObject(bufferB);
                clReleaseMemObject(bufferC);

                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);

                return 15;
            }

            t.nextLap(); // При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }

        // Среднее время круга (вычисления кернела) на самом деле считаются не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклониение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще) достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        
        // TODO 13 Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "GFlops: " << (n / t.lapAvg() / 1.0e9) << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти т.о. 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << ((double)n * 3 * sizeof(float) / t.lapAvg() / (1024.0 * 1024.0 * 1024.0)) << " GB/s" << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            // clEnqueueReadBuffer...
            for (int j = 0; j<n; j++) cs.data()[j] = 5.5;

            cl_int read_status = clEnqueueReadBuffer(queue, bufferC, 1, 0, n * sizeof(float), cs.data(), 0, nullptr, nullptr);

            if (read_status != CL_SUCCESS) {
                std::cerr << "error reading at iteration #" << i << ": " << read_status << std::endl;

                for (int i=0; i<n_kernels; i++) {
                    clReleaseKernel(kernels[i]);
                }


                clReleaseProgram(program);

                clReleaseMemObject(bufferA);
                clReleaseMemObject(bufferB);
                clReleaseMemObject(bufferC);

                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);

                return 16;
            }


            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << ((double)n * 2 /* одно чтение и одна записи */  * sizeof(float) / t.lapAvg() / (1024.0 * 1024.0 * 1024.0)) << " GB/s" << std::endl;
    }


    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            //throw std::runtime_error("CPU and GPU results differ at index ");
            std::cout << "CPU and GPU results differ at index " << i << std::endl;
            std::cout << as[i] << ", " << bs[i] << ", " << cs[i] << "    " << (cs[i] - (as[i] + bs[i])) << std::endl;

            throw std::runtime_error("CPU and GPU results differ");

        }
    }

    for (int i=0; i<n_kernels; i++) {
        clReleaseKernel(kernels[i]);
    }

    clReleaseProgram(program);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return 0;
}
