#include "face_dlib.h"

#include <iostream>

vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
);


Face_DLIB::Face_DLIB(QObject *parent) :
    QObject(parent)
{

}

Face_DLIB::~Face_DLIB()
{

}

vector<dlib::rectangle> Face_DLIB::get_face_rects(matrix<rgb_pixel> img)
{
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    std::vector<dlib::rectangle> face_rects = detector(img);
    return face_rects;
}

vector<matrix<rgb_pixel>> Face_DLIB::detect_faces(matrix<rgb_pixel> img, std::vector<dlib::rectangle> face_rects)
{
    std::vector<matrix<rgb_pixel>> faces;
    for (auto face_rect : face_rects)
    {
        auto shape = this->sp(img, face_rect);
        matrix<rgb_pixel> face_chip;
        dlib::extract_image_chip(img, dlib::get_face_chip_details(shape,150,0.25), face_chip);
        faces.push_back(std::move(face_chip));
    }
    return faces;
}

bool Face_DLIB::identify_faces(matrix<float,0,1> face_descriptor_src, matrix<float,0,1> face_descriptor_cmp, double faces_threshold)
{
    double length = double(dlib::length(face_descriptor_src - face_descriptor_cmp));
    return (length < faces_threshold);
}

double Face_DLIB::identify_faces_length(matrix<float,0,1> face_descriptor_src, matrix<float,0,1> face_descriptor_cmp)
{
    return double(dlib::length(face_descriptor_src - face_descriptor_cmp));
}

vector<matrix<float,0,1>> Face_DLIB::get_face_descriptors(vector<matrix<rgb_pixel>> faces)
{
    return this->net(faces);
}

matrix<float,0,1> Face_DLIB::get_face_descriptor_of_face(matrix<rgb_pixel> img)
{
    vector<matrix<rgb_pixel>> tmp_faces;
    tmp_faces.push_back(img);
    return (this->net(tmp_faces)[0]);
}

void Face_DLIB::set_net(string path_to_model_net)
{
    dlib::deserialize(path_to_model_net) >> this->net;
}

void Face_DLIB::set_sp(string path_to_model_sp)
{
    dlib::deserialize(path_to_model_sp) >> this->sp;
}

vector<matrix<rgb_pixel>> jitter_image(const matrix<rgb_pixel>& img)
{
    // All this function does is make 100 copies of img, all slightly jittered by being
    // zoomed, rotated, and translated a little bit differently. They are also randomly
    // mirrored left to right.
    thread_local dlib::rand rnd;

    vector<matrix<rgb_pixel>> crops;
    for (int i = 0; i < 100; ++i)
        crops.push_back(dlib::jitter_image(img,rnd));

    return crops;
}

void Face_DLIB::set_status(string status)
{
    emit status_changed(QString::fromStdString(status));
}
