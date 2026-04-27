/**
 * train_sp_vocabulary.cc
 *
 * Train a DBoW2 vocabulary tree from SuperPoint descriptors.
 * Reads descriptors from a binary file produced by extract_sp_descriptors.py,
 * trains a vocabulary tree, and saves to text file.
 *
 * Usage:
 *   ./train_sp_vocabulary <descriptors.bin> <output_vocabulary.txt> [k] [L]
 *
 * Parameters:
 *   k: branching factor (default: 10)
 *   L: depth levels (default: 5)
 *
 * Example:
 *   ./train_sp_vocabulary sp_descriptors.bin Vocabulary/SPvoc.txt 10 5
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdint>

#include <opencv2/core/core.hpp>

#include "DBoW2/FClass.h"
#include "DBoW2/FSuperPoint.h"
#include "DBoW2/TemplatedVocabulary.h"

using namespace std;

typedef DBoW2::TemplatedVocabulary<DBoW2::FSuperPoint::TDescriptor, DBoW2::FSuperPoint>
  SuperPointVocabulary;

bool loadDescriptors(const string &filename,
                     vector<vector<cv::Mat> > &features,
                     int max_per_image = 200,
                     int image_stride = 1)
{
    ifstream f(filename.c_str(), ios::binary);
    if (!f.is_open()) {
        cerr << "ERROR: Cannot open " << filename << endl;
        return false;
    }

    int32_t num_images;
    f.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));

    cout << "Reading " << num_images << " images from " << filename << endl;
    if (image_stride > 1)
        cout << "  Subsampling: every " << image_stride << "th image" << endl;
    if (max_per_image > 0)
        cout << "  Max features per image: " << max_per_image << endl;

    vector<int32_t> num_kp(num_images);
    for (int i = 0; i < num_images; i++) {
        f.read(reinterpret_cast<char*>(&num_kp[i]), sizeof(num_kp[i]));
    }

    features.resize(num_images);
    int total_kp = 0;
    int valid_images = 0;

    // Calculate file offsets for random access
    // Header: 4 bytes (num_images) + 4*num_images bytes (kp counts)
    size_t data_start = 4 + 4 * num_images;
    vector<size_t> image_offsets(num_images, 0);
    size_t offset = data_start;
    for (int i = 0; i < num_images; i++) {
        image_offsets[i] = offset;
        offset += num_kp[i] * 256 * sizeof(float);
    }

    for (int i = 0; i < num_images; i++) {
        // Subsample: skip images not in stride
        if (image_stride > 1 && (i % image_stride) != 0) {
            continue;
        }

        if (num_kp[i] == 0) {
            continue;
        }

        // Determine how many features to take from this image
        int n_take = num_kp[i];
        if (max_per_image > 0 && n_take > max_per_image) {
            // Take evenly spaced features
            n_take = max_per_image;
        }

        features[i].resize(n_take);

        // Read descriptors (possibly subsampled)
        if (n_take == num_kp[i]) {
            // Read all
            f.seekg(image_offsets[i]);
            vector<float> buf(num_kp[i] * 256);
            f.read(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(float));
            for (int j = 0; j < num_kp[i]; j++) {
                features[i][j].create(1, 256, CV_32F);
                memcpy(features[i][j].ptr<float>(), &buf[j * 256], 256 * sizeof(float));
            }
        } else {
            // Read all, then subsample
            f.seekg(image_offsets[i]);
            vector<float> buf(num_kp[i] * 256);
            f.read(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(float));
            float step = (float)num_kp[i] / n_take;
            for (int j = 0; j < n_take; j++) {
                int src_j = (int)(j * step);
                features[i][j].create(1, 256, CV_32F);
                memcpy(features[i][j].ptr<float>(), &buf[src_j * 256], 256 * sizeof(float));
            }
        }

        total_kp += n_take;
        valid_images++;

        if (valid_images <= 3 || valid_images % 50 == 0 || i == num_images - 1) {
            cout << "  Image " << i << ": " << n_take << "/" << num_kp[i] << " keypoints" << endl;
        }
    }

    // Remove empty entries
    vector<vector<cv::Mat> > features_clean;
    features_clean.reserve(valid_images);
    for (int i = 0; i < num_images; i++) {
        if (!features[i].empty()) {
            features_clean.push_back(features[i]);
        }
    }
    features.swap(features_clean);

    cout << "Total: " << total_kp << " keypoints from " << valid_images << " images" << endl;
    return true;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <descriptors.bin> <output.txt> [k] [L] [max_per_image] [image_stride]" << endl;
        cout << "  k: branching factor (default: 10)" << endl;
        cout << "  L: depth levels (default: 5)" << endl;
        cout << "  max_per_image: max features per image (default: 200)" << endl;
        cout << "  image_stride: use every Nth image (default: 1)" << endl;
        cout << "  Vocabulary size = k^L words" << endl;
        return 1;
    }

    string input_file = argv[1];
    string output_file = argv[2];
    int k = (argc > 3) ? atoi(argv[3]) : 10;
    int L = (argc > 4) ? atoi(argv[4]) : 5;
    int max_per_image = (argc > 5) ? atoi(argv[5]) : 200;
    int image_stride = (argc > 6) ? atoi(argv[6]) : 1;

    cout << "SuperPoint Vocabulary Training" << endl;
    cout << "  Input:    " << input_file << endl;
    cout << "  Output:   " << output_file << endl;
    cout << "  k=" << k << ", L=" << L << " (vocabulary size: " << (int)pow(k, L) << " words)" << endl;
    cout << endl;

    // Load descriptors
    vector<vector<cv::Mat> > features;
    if (!loadDescriptors(input_file, features, max_per_image, image_stride)) {
        return 1;
    }

    if (features.empty()) {
        cerr << "ERROR: No features loaded!" << endl;
        return 1;
    }

    // Count total features
    size_t total = 0;
    size_t valid_images = 0;
    for (size_t i = 0; i < features.size(); i++) {
        if (!features[i].empty()) {
            total += features[i].size();
            valid_images++;
        }
    }
    cout << "Valid images: " << valid_images << ", total features: " << total << endl;
    cout << endl;

    // Create vocabulary
    cout << "Training vocabulary (this may take a while)..." << endl;
    SuperPointVocabulary voc(k, L, DBoW2::TF_IDF, DBoW2::L1_NORM);

    // Use the SURF-style training which calls create internally
    clock_t t_start = clock();
    voc.create(features);
    clock_t t_end = clock();

    double elapsed = double(t_end - t_start) / CLOCKS_PER_SEC;
    cout << "Training completed in " << elapsed << " seconds" << endl;
    cout << "Vocabulary size: " << voc.size() << " words" << endl;
    cout << endl;

    // Save to text file
    cout << "Saving vocabulary to " << output_file << "..." << endl;
    voc.saveToTextFile(output_file);
    cout << "Vocabulary saved!" << endl;

    // Verify by loading back
    cout << "Verifying by loading back..." << endl;
    SuperPointVocabulary voc2;
    bool loaded = voc2.loadFromTextFile(output_file);
    if (loaded) {
        cout << "Verification PASSED! Loaded " << voc2.size() << " words" << endl;
    } else {
        cerr << "Verification FAILED!" << endl;
        return 1;
    }

    // Quick transform test on first image
    if (valid_images > 0) {
        DBoW2::BowVector bow;
        DBoW2::FeatureVector featVec;
        for (size_t i = 0; i < features.size(); i++) {
            if (!features[i].empty()) {
                cout << "Testing transform on image " << i << " (" << features[i].size() << " features)..." << endl;
                voc2.transform(features[i], bow, featVec, 4);
                cout << "  BoW vector: " << bow.size() << " nonzero entries" << endl;
                cout << "  Feature vector: " << featVec.size() << " nodes" << endl;
                break;
            }
        }
    }

    cout << endl << "All done!" << endl;
    return 0;
}
