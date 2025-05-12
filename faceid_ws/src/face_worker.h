#ifndef FACE_WORKER_H
#define FACE_WORKER_H

#include "face_dlib.h"
#include "face_mainwindow.h"
#include "face_structs.h"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/bind.hpp>
#include <dlib/dnn.h>
#include <dlib/opencv.h>
#include <dlib/sort.h>
#include <dlib/geometry.h>
#include <dlib/matrix.h>
#include <opencv2/opencv.hpp>
#include <QImage>
#include <QObject>
#include <QMutex>
#include <QTimer>
#include <QEventLoop>
#include <QApplication>
#include <QThread>
#include <QString>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <regex>
#include <string>

class Face_Worker : public QObject
{
    Q_OBJECT
public:
    explicit Face_Worker(QObject *parent = Q_NULLPTR);
    ~Face_Worker();

    void setup_vars(std::string path_to_file, std::string path_to_faces_database, std::string path_to_modelSP, std::string path_to_modelNet,
                    bool save_faces, double faces_threshold, double faces_voting_threshold, bool is_identifying_faces);
    void set_view_size(cv::Size size);
    void setup(Face_DLIB *fid, Face_Mainwindow *fmwin);
    void set_faceIDs(std::vector<FaceID> faceIDs);
    QEventLoop unknown_name_face_loop;
    std::vector<FaceID> get_faceIDs();

private:
    QMutex face_central_mutex;
    QMutex face_asked_mutex;
    Face_DLIB *fdlib;
    Face_Mainwindow *fmwin;
    std::vector<dlib::matrix<dlib::rgb_pixel>> faces;
    std::vector<FaceID> faceIDs;
    std::vector<Face_Data> faces_data;
    std::vector<dlib::sample_pair> edges;

    Face_Data* unknown_name_face;
    bool unknown_wait_until;

    Face_Data* center_face_data;

    std::string path_to_file;
    std::string path_to_faces_database;
    std::string path_to_modelNet;
    std::string path_to_modelSP;
    double faces_threshold;
    double faces_voting_threshold;

    bool is_identifying_faces;
    bool wait_until_got_face_name;
    bool learning_central_face;
    bool save_new_faces;

    size_t new_face_unknown_index;
    std::string central_face_new_name;

    cv::VideoCapture video;

    void start_analysis();
    void setup_models(std::string path_to_modelSP, std::string path_to_modelNet);

    cv::Size viewSize;

    void process_video();

    bool test_distance(dlib::point new_face_center, dlib::point old_face_center, double old_face_height, double old_face_width);

    void load_file(std::string fileName);
    void faces_save_identify(size_t frame_id);
    void faces_save_unknown(Face_Data faces_data_unknown);

    void faces_track(std::vector<struct Face_Data> new_faces_data);
    void faces_identify();
    double faces_identify_get_is_faceID(struct Face_Data face_data, std::string name);

    void faces_unknown_identify();
    void faces_add_faceID(Face_Data face_data);
    std::string faces_save_faceID(FaceID* faceID, bool check_name);
    void update_faceIDs();

    std::string generate_new_faceID(Face_Data face_data, bool unknown);
    void add_new_faces_to_unknown_faceID(std::vector<Face>);
    std::string rename_faceID(std::string face_name_new, std::string face_name_old);
    cv::Mat faces_paint(cv::Mat frame, double scaling);


    void learn_central_face();
    void faces_set_center(cv::Mat frame);
    FaceID* search_faceID(std::string name);

    bool test_is_new(std::vector<Face> faces, dlib::matrix<float,0,1> face_descriptor);

    cv::Mat resize_image(cv::Mat src, double width, double height);
    QImage mat_to_QImage(cv::Mat src);
    cv::Rect dlib_rect_to_cv_rect(dlib::rectangle drct);
    cv::Rect resize_rect(cv::Rect rct, double x);
    void set_status(std::string status);
    void view_image(cv::Mat src);
    cv::Rect change_rect(dlib::rectangle drct);
    cv::Rect scale_rect(cv::Rect rct, double scale);
    cv::Mat scale_image(cv::Mat src, double scale);
    cv::Mat load_image(std::string path_to_image);

    bool compare_mat(cv::Mat img1, cv::Mat img2);

    void took_time(std::string title, std::chrono::high_resolution_clock::time_point timestamp1, std::chrono::high_resolution_clock::time_point timestamp2);

signals:
    void finished();
    void error(QString err);
    void status_changed(QString status);
    void image_changed(QImage qimg);
    void asked_new_name();
    void set_button_learn(bool active);
    void window_closed();

public slots:
    void process();
    void change_QStatus(QString qStatus);
    void set_learn_central_face(QString central_face_new_name);
    void face_unknown_identified();
    void close_window();
};

#endif // FACE_WORKER_H
