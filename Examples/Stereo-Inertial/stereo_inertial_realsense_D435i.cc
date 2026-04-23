/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include <signal.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>

#include <condition_variable>

#include <opencv2/core/core.hpp>
#include <thread> // 必须添加，否则编译会报错找不到 thread
#include <librealsense2/rs.hpp>
#include "librealsense2/rsutil.h"

#include <System.h>

using namespace std;

bool b_continue_session;

void exit_loop_handler(int s){
    cout << "Finishing session" << endl;
    b_continue_session = false;

}

void interpolateData(const std::vector<double> &vBase_times,
                     std::vector<rs2_vector> &vInterp_data, std::vector<double> &vInterp_times,
                     const rs2_vector &prev_data, const double &prev_time);

rs2_vector interpolateMeasure(const double target_time,
                              const rs2_vector current_data, const double current_time,
                              const rs2_vector prev_data, const double prev_time);

static rs2_option get_sensor_option(const rs2::sensor& sensor)
{
    // Sensors usually have several options to control their properties
    //  such as Exposure, Brightness etc.

    std::cout << "Sensor supports the following options:\n" << std::endl;

    // The following loop shows how to iterate over all available options
    // Starting from 0 until RS2_OPTION_COUNT (exclusive)
    for (int i = 0; i < static_cast<int>(RS2_OPTION_COUNT); i++)
    {
        rs2_option option_type = static_cast<rs2_option>(i);
        //SDK enum types can be streamed to get a string that represents them
        std::cout << "  " << i << ": " << option_type;

        // To control an option, use the following api:

        // First, verify that the sensor actually supports this option
        if (sensor.supports(option_type))
        {
            std::cout << std::endl;

            // Get a human readable description of the option
            const char* description = sensor.get_option_description(option_type);
            std::cout << "       Description   : " << description << std::endl;

            // Get the current value of the option
            float current_value = sensor.get_option(option_type);
            std::cout << "       Current Value : " << current_value << std::endl;

            //To change the value of an option, please follow the change_sensor_option() function
        }
        else
        {
            std::cout << " is not supported" << std::endl;
        }
    }

    uint32_t selected_sensor_option = 0;
    return static_cast<rs2_option>(selected_sensor_option);
}

int main(int argc, char **argv) {

    if (argc < 3 || argc > 4) {
        cerr << endl
             << "Usage: ./stereo_inertial_realsense_D435i path_to_vocabulary path_to_settings (trajectory_file_name)"
             << endl;
        return 1;
    }

    string file_name;

    if (argc == 4) {
        file_name = string(argv[argc - 1]);
    }

    // FPS monitoring variables
    int frame_count = 0;
    double current_time = 0;
    double last_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    double fps_update_interval = 1.0; // Update FPS every second

    // Latency measurement variables
    double last_frame_hw_time = 0.0;           // hardware timestamp of previous frame
    std::vector<double> v_latency_grab_delay;   // time from hw timestamp to start of processing
    std::vector<double> v_latency_track;        // time for TrackStereo call
    std::vector<double> v_latency_total;        // total grab+track (end-to-end SLAM delay)
    std::vector<double> v_frame_interval;       // actual interval between consecutive frames
    std::chrono::steady_clock::time_point last_process_end_tp;  // track when last frame finished processing

    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = exit_loop_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, NULL);
    b_continue_session = true;

    double offset = 0; // ms

    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    rs2::device selected_device;
    if (devices.size() == 0)
    {
        std::cerr << "No device connected, please connect a RealSense device" << std::endl;
        return 0;
    }
    else
        selected_device = devices[0];

    std::cout << "========================================" << std::endl;
    std::cout << "ORB-SLAM3 Stereo-Inertial D435i @ 30 FPS" << std::endl;
    std::cout << "========================================" << std::endl;

    std::vector<rs2::sensor> sensors = selected_device.query_sensors();
    int index = 0;
    // We can now iterate the sensors and print their names
    for (rs2::sensor sensor : sensors)
        if (sensor.supports(RS2_CAMERA_INFO_NAME)) {
            ++index;
            if (index == 1) {
                sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
                sensor.set_option(RS2_OPTION_AUTO_EXPOSURE_LIMIT,10000);  // Reduced for 30 FPS
                sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1); // switch ON emitter for better depth sensing
                sensor.set_option(RS2_OPTION_FRAMES_QUEUE_SIZE, 16);  // Queue size for 30 FPS
            }
            // std::cout << "  " << index << " : " << sensor.get_info(RS2_CAMERA_INFO_NAME) << std::endl;
            get_sensor_option(sensor);
            if (index == 2){
                // RGB camera (not used here...)
                sensor.set_option(RS2_OPTION_EXPOSURE,100.f);
            }

            if (index == 3){
                sensor.set_option(RS2_OPTION_ENABLE_MOTION_CORRECTION,0);
            }

        }

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Create a configuration for configuring the pipeline with a non default profile
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, 30);  // 30 FPS
    cfg.enable_stream(RS2_STREAM_INFRARED, 2, 640, 480, RS2_FORMAT_Y8, 30);  // 30 FPS
    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);

    // IMU callback
    std::mutex imu_mutex;
    std::condition_variable cond_image_rec;

    vector<double> v_accel_timestamp;
    vector<rs2_vector> v_accel_data;
    vector<double> v_gyro_timestamp;
    vector<rs2_vector> v_gyro_data;

    double prev_accel_timestamp = 0;
    rs2_vector prev_accel_data;
    double current_accel_timestamp = 0;
    rs2_vector current_accel_data;
    vector<double> v_accel_timestamp_sync;
    vector<rs2_vector> v_accel_data_sync;

    cv::Mat imCV, imRightCV;
    int width_img, height_img;
    double timestamp_image = -1.0;
    bool image_ready = false;
    int count_im_buffer = 0; // count dropped frames

    // IMU buffer monitoring
    int imu_starvation_count = 0;  // count frames with too few IMU samples

    auto imu_callback = [&](const rs2::frame& frame)
    {
        std::unique_lock<std::mutex> lock(imu_mutex);

        if(rs2::frameset fs = frame.as<rs2::frameset>())
        {
            count_im_buffer++;

            double new_timestamp_image = fs.get_timestamp()*1e-3;
            if(abs(timestamp_image-new_timestamp_image)<0.001){
                // cout << "Two frames with the same timeStamp!!!\n";
                count_im_buffer--;
                return;
            }

            rs2::video_frame ir_frameL = fs.get_infrared_frame(1);
            rs2::video_frame ir_frameR = fs.get_infrared_frame(2);

            imCV = cv::Mat(cv::Size(width_img, height_img), CV_8U, (void*)(ir_frameL.get_data()), cv::Mat::AUTO_STEP);
            imRightCV = cv::Mat(cv::Size(width_img, height_img), CV_8U, (void*)(ir_frameR.get_data()), cv::Mat::AUTO_STEP);

            timestamp_image = fs.get_timestamp()*1e-3;
            image_ready = true;

            while(v_gyro_timestamp.size() > v_accel_timestamp_sync.size())
            {

                int index = v_accel_timestamp_sync.size();
                double target_time = v_gyro_timestamp[index];

                rs2_vector interp_data = interpolateMeasure(target_time, current_accel_data, current_accel_timestamp,
                                                            prev_accel_data, prev_accel_timestamp);
                v_accel_data_sync.push_back(interp_data);
                v_accel_timestamp_sync.push_back(target_time);
            }

            lock.unlock();
            cond_image_rec.notify_all();
        }
        else if (rs2::motion_frame m_frame = frame.as<rs2::motion_frame>())
        {
            if (m_frame.get_profile().stream_name() == "Gyro")
            {
                // It runs at 200Hz
                v_gyro_data.push_back(m_frame.get_motion_data());
                v_gyro_timestamp.push_back((m_frame.get_timestamp()+offset)*1e-3);
                //rs2_vector gyro_sample = m_frame.get_motion_data();
                //std::cout << "Gyro:" << gyro_sample.x << ", " << gyro_sample.y << ", " << gyro_sample.z << std::endl;
            }
            else if (m_frame.get_profile().stream_name() == "Accel")
            {
                // It runs at 60Hz
                prev_accel_timestamp = current_accel_timestamp;
                prev_accel_data = current_accel_data;

                current_accel_data = m_frame.get_motion_data();
                current_accel_timestamp = (m_frame.get_timestamp()+offset)*1e-3;

                while(v_gyro_timestamp.size() > v_accel_timestamp_sync.size())
                {
                    int index = v_accel_timestamp_sync.size();
                    double target_time = v_gyro_timestamp[index];

                    rs2_vector interp_data = interpolateMeasure(target_time, current_accel_data, current_accel_timestamp,
                                                                prev_accel_data, prev_accel_timestamp);

                    v_accel_data_sync.push_back(interp_data);
                    v_accel_timestamp_sync.push_back(target_time);

                }
                // std::cout << "Accel:" << current_accel_data.x << ", " << current_accel_data.y << ", " << current_accel_data.z << std::endl;
            }
        }
    };

    rs2::pipeline_profile pipe_profile = pipe.start(cfg, imu_callback);

// --- 新增：给 D435i 硬件 2 秒钟的“起床时间”来加载标定表 ---
    std::cout << "Pipeline started. Waiting for hardware to stabilize..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    // -------------------------------------------------------

    vector<ORB_SLAM3::IMU::Point> vImuMeas;
    rs2::stream_profile cam_left = pipe_profile.get_stream(RS2_STREAM_INFRARED, 1);
    rs2::stream_profile cam_right = pipe_profile.get_stream(RS2_STREAM_INFRARED, 2);


    rs2::stream_profile imu_stream = pipe_profile.get_stream(RS2_STREAM_GYRO);
    float* Rbc = cam_left.get_extrinsics_to(imu_stream).rotation;
    float* tbc = cam_left.get_extrinsics_to(imu_stream).translation;
    
    
    // --- 插入以下内容 ---
// 强行纠正 Tbc (相机到 IMU 的位移，D435i 标准值约为 15mm)
    tbc[0] = -0.015f; tbc[1] = 0.0f; tbc[2] = 0.0f; 
// 强行纠正 Rbc (旋转矩阵设为单位阵)
    for(int i=0; i<9; i++) Rbc[i] = (i%4==0) ? 1.0f : 0.0f;
// ------------------

    std::cout << "Tbc (left) = " << std::endl;
    for(int i = 0; i<3; i++){
        for(int j = 0; j<3; j++)
            std::cout << Rbc[i*3 + j] << ", ";
        std::cout << tbc[i] << "\n";
    }

    float* Rlr = cam_right.get_extrinsics_to(cam_left).rotation;
    float* tlr = cam_right.get_extrinsics_to(cam_left).translation;
    
    
    // 强行纠正 Tlr (左右相机基线，D435i 标准值为 50mm)
   tlr[0] = -0.05f; tlr[1] = 0.0f; tlr[2] = 0.0f;
// 强行纠正 Rlr (旋转矩阵设为单位阵)
   for(int i=0; i<9; i++) Rlr[i] = (i%4==0) ? 1.0f : 0.0f;

   std::cout << "!!! EXPLICIT FIX APPLIED: Standard D435i Extrinsics Loaded !!!" << std::endl;
// ------------------

    std::cout << "Tlr  = " << std::endl;
    for(int i = 0; i<3; i++){
        for(int j = 0; j<3; j++)
            std::cout << Rlr[i*3 + j] << ", ";
        std::cout << tlr[i] << "\n";
    }



    rs2_intrinsics intrinsics_left = cam_left.as<rs2::video_stream_profile>().get_intrinsics();
    width_img = intrinsics_left.width;
    height_img = intrinsics_left.height;
    cout << "Left camera: \n";
    std::cout << " fx = " << intrinsics_left.fx << std::endl;
    std::cout << " fy = " << intrinsics_left.fy << std::endl;
    std::cout << " cx = " << intrinsics_left.ppx << std::endl;
    std::cout << " cy = " << intrinsics_left.ppy << std::endl;
    std::cout << " height = " << intrinsics_left.height << std::endl;
    std::cout << " width = " << intrinsics_left.width << std::endl;
    std::cout << " Coeff = " << intrinsics_left.coeffs[0] << ", " << intrinsics_left.coeffs[1] << ", " <<
        intrinsics_left.coeffs[2] << ", " << intrinsics_left.coeffs[3] << ", " << intrinsics_left.coeffs[4] << ", " << std::endl;
    std::cout << " Model = " << intrinsics_left.model << std::endl;

    rs2_intrinsics intrinsics_right = cam_right.as<rs2::video_stream_profile>().get_intrinsics();
    width_img = intrinsics_right.width;
    height_img = intrinsics_right.height;
    cout << "Right camera: \n";
    std::cout << " fx = " << intrinsics_right.fx << std::endl;
    std::cout << " fy = " << intrinsics_right.fy << std::endl;
    std::cout << " cx = " << intrinsics_right.ppx << std::endl;
    std::cout << " cy = " << intrinsics_right.ppy << std::endl;
    std::cout << " height = " << intrinsics_right.height << std::endl;
    std::cout << " width = " << intrinsics_right.width << std::endl;
    std::cout << " Coeff = " << intrinsics_right.coeffs[0] << ", " << intrinsics_right.coeffs[1] << ", " <<
        intrinsics_right.coeffs[2] << ", " << intrinsics_right.coeffs[3] << ", " << intrinsics_right.coeffs[4] << ", " << std::endl;
    std::cout << " Model = " << intrinsics_right.model << std::endl;


    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::IMU_STEREO, true, 0, file_name);
    float imageScale = SLAM.GetImageScale();

    double timestamp;
    cv::Mat im, imRight;

    // Clear IMU vectors
    v_gyro_data.clear();
    v_gyro_timestamp.clear();
    v_accel_data_sync.clear();
    v_accel_timestamp_sync.clear();

    double t_resize = 0.f;
    double t_track = 0.f;

    // Latency output file
    std::ofstream latency_log;
    latency_log.open("slam_latency_log.csv");
    latency_log << "frame_id,hw_timestamp,grab_delay_ms,track_delay_ms,total_delay_ms,frame_interval_ms,num_imu_samples\n";

    while (!SLAM.isShutDown())
    {
        std::vector<rs2_vector> vGyro;
        std::vector<double> vGyro_times;
        std::vector<rs2_vector> vAccel;
        std::vector<double> vAccel_times;

        {
            std::unique_lock<std::mutex> lk(imu_mutex);
            if(!image_ready)
                cond_image_rec.wait(lk);

#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point time_Start_Process = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point time_Start_Process = std::chrono::monotonic_clock::now();
#endif

            if(count_im_buffer>1)
                cout << count_im_buffer -1 << " dropped frs\n";
            count_im_buffer = 0;

            // IMU buffer starvation check
            // At 30 FPS with 200Hz gyro, expect ~6-7 IMU samples per frame minimum
            if (v_gyro_data.size() < 1) {
                imu_starvation_count++;
                if (imu_starvation_count <= 10 || imu_starvation_count % 100 == 0)
                    cout << "[WARNING] IMU starvation! Only " << v_gyro_data.size()
                         << " gyro samples for this frame (starved " << imu_starvation_count << " times total)\n";
            }

            while(v_gyro_timestamp.size() > v_accel_timestamp_sync.size())
            {
                int index = v_accel_timestamp_sync.size();
                double target_time = v_gyro_timestamp[index];

                rs2_vector interp_data = interpolateMeasure(target_time, current_accel_data, current_accel_timestamp, prev_accel_data, prev_accel_timestamp);

                v_accel_data_sync.push_back(interp_data);
                // v_accel_data_sync.push_back(current_accel_data); // 0 interpolation
                v_accel_timestamp_sync.push_back(target_time);
            }

            // Copy the IMU data
            vGyro = v_gyro_data;
            vGyro_times = v_gyro_timestamp;
            vAccel = v_accel_data_sync;
            vAccel_times = v_accel_timestamp_sync;
            timestamp = timestamp_image;
            im = imCV.clone();
            imRight = imRightCV.clone();

            // Clear IMU vectors
            v_gyro_data.clear();
            v_gyro_timestamp.clear();
            v_accel_data_sync.clear();
            v_accel_timestamp_sync.clear();

            image_ready = false;
        }


        for(int i=0; i<vGyro.size(); ++i)
        {
            ORB_SLAM3::IMU::Point lastPoint(vAccel[i].x, vAccel[i].y, vAccel[i].z,
                                  vGyro[i].x, vGyro[i].y, vGyro[i].z,
                                  vGyro_times[i]);
            vImuMeas.push_back(lastPoint);
        }

        // Skip frames with no IMU data — feeding empty IMU to TrackStereo causes
        // "not IMU meas" → StereoInitialization fails → map reset loop
        if (vImuMeas.empty() && vGyro.size() == 0) {
            cout << "[SKIP] No IMU data for frame (timestamp=" << fixed << setprecision(3) << timestamp << "), skipping" << endl;
            vImuMeas.clear();
            continue;
        }
        if (vImuMeas.size() < 2) {
            // Need at least 2 IMU measurements for PreintegrateIMU to work (n-1 integration steps)
            cout << "[SKIP] Only " << vImuMeas.size() << " IMU samples (need >=2), frame_ts=" << timestamp
                 << " imu_ts=[" << setprecision(3) << vImuMeas.front().t << " - " << vImuMeas.back().t << "]" << endl;
            vImuMeas.clear();
            continue;
        }

        // Debug: print IMU time range vs frame timestamp
        static int debug_print_count = 0;
        if (debug_print_count < 20) {
            cout << "[DEBUG] frame_ts=" << fixed << setprecision(3) << timestamp
                 << " imu_count=" << vImuMeas.size()
                 << " imu_ts=[" << vImuMeas.front().t << " - " << vImuMeas.back().t << "]" << endl;
            debug_print_count++;
        }

        // --- Latency: measure grab delay ---
        // grab_delay = time spent waiting for this frame (from last frame processed to now)
        double grab_delay_ms = 0.0;
        {
            auto now_tp = std::chrono::steady_clock::now();
            if (last_process_end_tp.time_since_epoch().count() != 0) {
                grab_delay_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                    now_tp - last_process_end_tp).count();
            }
        }

        if(imageScale != 1.f)
        {
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t_Start_Resize = std::chrono::monotonic_clock::now();
    #endif
#endif
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
            cv::resize(imRight, imRight, cv::Size(width, height));

#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t_End_Resize = std::chrono::monotonic_clock::now();
    #endif
            t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
            SLAM.InsertResizeTime(t_resize);
#endif
        }

#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t_Start_Track = std::chrono::steady_clock::now();
    #else
        std::chrono::monotonic_clock::time_point t_Start_Track = std::chrono::monotonic_clock::now();
    #endif
#endif
        // Stereo images are already rectified.
        SLAM.TrackStereo(im, imRight, timestamp, vImuMeas);
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t_End_Track = std::chrono::steady_clock::now();
    #else
        std::chrono::monotonic_clock::time_point t_End_Track = std::chrono::monotonic_clock::now();
    #endif
        t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Track - t_Start_Track).count();
        SLAM.InsertTrackTime(t_track);
#endif

        // --- Latency: compute total delay ---
        double track_delay_ms = t_track;  // already in ms from REGISTER_TIMES
        double total_delay_ms = grab_delay_ms + track_delay_ms;

        // Frame interval
        double frame_interval_ms = 0.0;
        if (last_frame_hw_time > 0) {
            frame_interval_ms = (timestamp - last_frame_hw_time) * 1000.0;
        }
        last_frame_hw_time = timestamp;

        // Store for statistics
        v_latency_grab_delay.push_back(grab_delay_ms);
        v_latency_track.push_back(track_delay_ms);
        v_latency_total.push_back(total_delay_ms);
        v_frame_interval.push_back(frame_interval_ms);

        // Log to CSV
        static int log_frame_id = 0;
        latency_log << log_frame_id++ << ","
                    << fixed << setprecision(3) << timestamp << ","
                    << grab_delay_ms << ","
                    << track_delay_ms << ","
                    << total_delay_ms << ","
                    << frame_interval_ms << ","
                    << vGyro.size() << "\n";

        // FPS monitoring + latency statistics (every 2 seconds)
        frame_count++;
        current_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        if (current_time - last_time >= 2.0) {
            double fps = frame_count / (current_time - last_time);
            double avg_grab = 0, avg_track = 0, avg_total = 0, max_total = 0;
            for (size_t i = 0; i < v_latency_total.size(); i++) {
                avg_grab += v_latency_grab_delay[i];
                avg_track += v_latency_track[i];
                avg_total += v_latency_total[i];
                if (v_latency_total[i] > max_total) max_total = v_latency_total[i];
            }
            if (!v_latency_total.empty()) {
                avg_grab /= v_latency_total.size();
                avg_track /= v_latency_total.size();
                avg_total /= v_latency_total.size();
            }
            double avg_interval = 0;
            for (size_t i = 0; i < v_frame_interval.size(); i++) avg_interval += v_frame_interval[i];
            if (!v_frame_interval.empty()) avg_interval /= v_frame_interval.size();

            cout << "[FPS: " << fixed << setprecision(1) << fps
                 << " | Avg grab: " << setprecision(1) << avg_grab << "ms"
                 << " | Avg track: " << avg_track << "ms"
                 << " | Avg total: " << avg_total << "ms"
                 << " | Max total: " << max_total << "ms"
                 << " | Avg interval: " << avg_interval << "ms"
                 << " | Frames: " << frame_count << "]" << endl;

            frame_count = 0;
            last_time = current_time;
        }



        // Clear the previous IMU measurements to load the new ones
        vImuMeas.clear();
        last_process_end_tp = std::chrono::steady_clock::now();
    }

    // Print latency summary
    latency_log.close();
    cout << "\n========== SLAM Latency Summary ==========" << endl;
    if (!v_latency_total.empty()) {
        double avg_grab=0, avg_track=0, avg_total=0, max_total=0, min_total=1e9;
        for (size_t i = 0; i < v_latency_total.size(); i++) {
            avg_grab += v_latency_grab_delay[i];
            avg_track += v_latency_track[i];
            avg_total += v_latency_total[i];
            max_total = std::max(max_total, v_latency_total[i]);
            min_total = std::min(min_total, v_latency_total[i]);
        }
        avg_grab /= v_latency_total.size();
        avg_track /= v_latency_total.size();
        avg_total /= v_latency_total.size();
        cout << "  Frames processed: " << v_latency_total.size() << endl;
        cout << "  Avg grab delay:    " << fixed << setprecision(2) << avg_grab << " ms" << endl;
        cout << "  Avg track time:    " << avg_track << " ms" << endl;
        cout << "  Avg total delay:   " << avg_total << " ms" << endl;
        cout << "  Min total delay:   " << min_total << " ms" << endl;
        cout << "  Max total delay:   " << max_total << " ms" << endl;

        // Percentile
        std::sort(v_latency_total.begin(), v_latency_total.end());
        size_t p50 = v_latency_total.size() * 50 / 100;
        size_t p90 = v_latency_total.size() * 90 / 100;
        size_t p95 = v_latency_total.size() * 95 / 100;
        size_t p99 = v_latency_total.size() * 99 / 100;
        cout << "  P50 total delay:   " << v_latency_total[p50] << " ms" << endl;
        cout << "  P90 total delay:   " << v_latency_total[p90] << " ms" << endl;
        cout << "  P95 total delay:   " << v_latency_total[p95] << " ms" << endl;
        cout << "  P99 total delay:   " << v_latency_total[p99] << " ms" << endl;
        cout << "  Detailed log:      slam_latency_log.csv" << endl;
    }
    cout << "==========================================\n" << endl;

    cout << "System shutdown!\n";
}

rs2_vector interpolateMeasure(const double target_time,
                              const rs2_vector current_data, const double current_time,
                              const rs2_vector prev_data, const double prev_time)
{

    // If there are not previous information, the current data is propagated
    if(prev_time == 0)
    {
        return current_data;
    }

    rs2_vector increment;
    rs2_vector value_interp;

    if(target_time > current_time) {
        value_interp = current_data;
    }
    else if(target_time > prev_time)
    {
        increment.x = current_data.x - prev_data.x;
        increment.y = current_data.y - prev_data.y;
        increment.z = current_data.z - prev_data.z;

        double factor = (target_time - prev_time) / (current_time - prev_time);

        value_interp.x = prev_data.x + increment.x * factor;
        value_interp.y = prev_data.y + increment.y * factor;
        value_interp.z = prev_data.z + increment.z * factor;
    }
    else {
        value_interp = prev_data;
    }

    return value_interp;
}
