#ifndef FACE_STRUCTS_H
#define FACE_STRUCTS_H

#include <dlib/geometry.h>
#include <string>
#include <iostream>

struct Face
{
    dlib::matrix<dlib::rgb_pixel> image;
    dlib::matrix<float,0,1> descriptor;
    bool is_saved;
};

struct FaceID
{
    std::string name;
    std::vector<struct Face> faces;
    bool is_saved = true;

    friend std::ostream& operator<<(std::ostream& os, const FaceID& faceID)
    {
        os << faceID.name << " (" << faceID.faces.size() << ")";
        return os;
    }
};


struct Face_Data
{
    std::string name;
    dlib::rectangle face_rect;
    dlib::point center;
    Face face;
    std::vector<Face> faces;
    size_t tracked_frames;
    bool is_tracked = false;
    bool is_center = false;
    bool is_new = false;
    std::vector<double> faces_voting;
    double is_faceID;
    bool is_unknown = false;

    friend void operator++(Face_Data& ihs)
    {
        ihs.tracked_frames++;
    }

    friend bool operator==(const Face_Data& lhs, const Face_Data& rhs)
    {
        return (lhs.name == rhs.name);
    }

    friend bool operator!=(const Face_Data& lhs, const Face_Data& rhs)
    {
        return (lhs.name == rhs.name);
    }

    friend std::ostream& operator<<(std::ostream& os, const Face_Data& face_data)
    {
        os << face_data.name << " (" << face_data.is_faceID << ")";
        return os;
    }

};
#endif // FACE_STRUCTS_H
