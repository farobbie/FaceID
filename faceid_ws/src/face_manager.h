#ifndef FACE_MANAGER_H
#define FACE_MANAGER_H

#include "face_dlib.h"
#include "face_mainwindow.h"
#include "face_worker.h"
#include "face_structs.h"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>
#include <dlib/dnn.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <QImage>
#include <QObject>
#include <QThread>
#include <QTimer>
#include <QEventLoop>
#include <QTime>
#include <QElapsedTimer>
#include <QString>
#include <string>
#include <tuple>

using std::string;
using std::vector;
using dlib::matrix;
using dlib::rgb_pixel;

class Face_Manager : public QObject
{
    Q_OBJECT

public:
    explicit Face_Manager(QObject *parent = Q_NULLPTR);
    ~Face_Manager();
    void setup(std::string path_to_file, std::string path_to_faces_database, std::string path_to_model_net, std::string path_to_modelSP,
               bool save_faces, double faces_threshold, double faces_voting_threshold);
    void start_process();
    void load_database(string path_to_faces_database);

private:
    Face_Mainwindow fmwin;
    Face_DLIB fdlib;
    Face_Worker face_worker;
    std::vector<FaceID> faceIDs;

    QThread* thread_worker;
    QThread* thread_runtime;
    QTimer* timer_runtime;
    QElapsedTimer timer_eruntime;
    void start_clock();
    bool create_new_face_folder(matrix<rgb_pixel> face_img, matrix<float,0,1> face_descriptor, string name);
    double compare_faces(matrix<float,0,1> face_descriptor, matrix<float,0,1> face_descriptor_cmp);
    bool generate_faceID(string name, Face face);
    bool generate_faceIDs(string name, vector<Face> faces);

public slots:

private slots:
    void update_clock();
    void close_application();

signals:
    void clock_updated(QString qtime);
};

#endif // FACE_MANAGER_H
