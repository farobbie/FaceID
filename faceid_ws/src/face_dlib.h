#ifndef FACE_DLIB_H
#define FACE_DLIB_H


#include <dlib/dnn.h>
#include <dlib/geometry.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <QObject>
#include <QString>


#include <tuple>
#include <string>

// ----------------------------------------------------------------------------------------

using std::vector;
using std::string;
using std::cout;
using std::endl;
using dlib::sample_pair;
using dlib::matrix;
using dlib::rectangle;
using dlib::rgb_pixel;

namespace dlib {
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;
}
// ----------------------------------------------------------------------------------------

class Face_DLIB : public QObject
{
    Q_OBJECT
public:
    explicit Face_DLIB(QObject *parent = Q_NULLPTR);
    ~Face_DLIB();

    vector<rectangle> get_face_rects(matrix<rgb_pixel> img);
    vector<matrix<rgb_pixel>> detect_faces(matrix<rgb_pixel> img, vector<rectangle> face_rects);
    bool identify_faces(matrix<float,0,1> face_descriptor_src, matrix<float,0,1> face_descriptor_cmp, double faces_threshold=0.6);
    double identify_faces_length(matrix<float,0,1> face_descriptor_src, matrix<float,0,1> face_descriptor_cmp);


    void set_net(string path_to_modelNet);
    void set_sp(string path_to_modelSp);
    std::vector<matrix<float,0,1>> get_face_descriptors(vector<matrix<rgb_pixel>> faces);
    matrix<float,0,1> get_face_descriptor_of_face(matrix<rgb_pixel> img);

private:
    dlib::anet_type net;
    dlib::shape_predictor sp;
    void set_status(string status);
signals:
    void status_changed(QString status);
};

#endif // FACE_DLIB_H
