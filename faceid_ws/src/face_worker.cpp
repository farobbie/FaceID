#include "face_worker.h"

#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>

using cclock = std::chrono::high_resolution_clock;
using cv::Mat;
using cv::Scalar;
using dlib::matrix;
using dlib::rgb_pixel;
using std::vector;
using std::string;
using std::endl;
using std::cout;
using std::stringstream;
using std::ostringstream;

Face_Worker::Face_Worker(QObject *parent) : QObject(parent)
{
}

Face_Worker::~Face_Worker()
{
}

void Face_Worker::setup(Face_DLIB *fdlib, Face_Mainwindow *fmwin)
{
    this->fdlib = fdlib;
    this->fmwin = fmwin;

    connect(this->fdlib, &Face_DLIB::status_changed, this, &Face_Worker::change_QStatus);
    connect(this->fmwin, &Face_Mainwindow::face_identified, this, &Face_Worker::face_unknown_identified);
}

void Face_Worker::set_faceIDs(std::vector<FaceID> faceIDs)
{
    this->faceIDs = faceIDs;
}

std::vector<FaceID> Face_Worker::get_faceIDs()
{
    return this->faceIDs;
}

void Face_Worker::setup_vars(string path_to_file, string path_to_faces_database, string path_to_modelNet, string path_to_modelSP,
                             bool save_faces, double faces_threshold, double faces_voting_threshold, bool identify_faces)
{
    this->save_new_faces = save_faces;
    this->path_to_file = path_to_file;

    this->path_to_modelNet = path_to_modelNet;
    this->path_to_modelSP = path_to_modelSP;
    this->path_to_faces_database = path_to_faces_database;
    this->faces_threshold = faces_threshold;
    this->faces_voting_threshold = faces_voting_threshold;
    this->is_identifying_faces = identify_faces;
    this->new_face_unknown_index = 0;

    this->center_face_data = nullptr;

    this->learning_central_face = false;
    this->wait_until_got_face_name = false;
}

void Face_Worker::process()
{
    this->start_analysis();
    emit finished();
}


void Face_Worker::start_analysis()
{
    this->load_file(this->path_to_file);
}

void Face_Worker::load_file(string filename)
{
    this->video.open(filename);
    this->process_video();
}


void Face_Worker::process_video()
{
    Mat frame;
    size_t frame_id = 0;
    ostringstream frameSS;
    this->faces_data.clear();

    cclock::time_point file_t1 = cclock::now();
    while(this->video.isOpened())
    {
        cclock::time_point frame_t1 = cclock::now();
        std::cout << "-----------------------------\n";
        this->video >> frame;

        if(frame_id >= 0 && !frame.empty())
        {
            frameSS.str(std::string());
            frameSS.clear();
            frameSS << "Analyse frame #" << frame_id << " of file";
            std::cout << frameSS.str() << "\n";
            this->set_status(frameSS.str());
            matrix<rgb_pixel> dlib_frame;
            assign_image(dlib_frame, dlib::cv_image<dlib::bgr_pixel>(frame));

            double scale_width = double(this->viewSize.width-2) / double(frame.cols);
            double scale_height = double(this->viewSize.height-2) / double(frame.rows);
            double scale_img = std::min(scale_width, scale_height);

            cv::Mat frame_resized = this->scale_image(frame, scale_img);

            vector<dlib::rectangle> face_rects;
            vector<matrix<rgb_pixel>> face_imgs;

            face_rects = this->fdlib->get_face_rects(dlib_frame);
            ostringstream dff;

            size_t num_faces = face_rects.size();
            if(num_faces > 0)
            {
                frameSS << " - found " << num_faces << " Faces";
                this->set_status(frameSS.str());

                face_imgs = this->fdlib->detect_faces(dlib_frame, face_rects);

                vector<matrix<float,0,1>> face_descriptors = this->fdlib->get_face_descriptors(face_imgs);

                //initialize new faces found in frame
                this->center_face_data = nullptr;
                vector<Face_Data> new_faces_data;
                ostringstream unknown_id;
                for(size_t n = 0; n < num_faces; ++n)
                {
                    auto &face_rect_tmp = face_rects[n];
                    Face_Data face_data;

                    face_data.name = "unknown";
                    face_data.face = Face {face_imgs[n], face_descriptors[n], false};
                    face_data.face_rect = face_rect_tmp;
                    face_data.center = face_rect_tmp.tl_corner() + (face_rect_tmp.br_corner() - face_rect_tmp.tl_corner()) * 0.5;
                    face_data.is_tracked = false;
                    face_data.is_new = true;
                    face_data.is_center = false;
                    face_data.is_unknown = false;
                    face_data.tracked_frames = 0;
                    new_faces_data.push_back(face_data);
                }

                this->faces_set_center(frame);
                //track faces
                this->faces_track(new_faces_data);

                //identify faces
                if(this->is_identifying_faces)
                {
                    this->faces_unknown_identify();
                    this->faces_identify();
                }


                this->update_faceIDs();

                //save new faces of current frame
                if(this->path_to_faces_database != "" && this->save_new_faces)
                {
                    //Save all faces from current frame in #path_to_database#/unknown/#filename#/
                    for(auto face_data : this->faces_data)
                    {
                        this->faces_save_unknown(face_data);
                    }
                }
                //paint faces
                frame_resized = this->faces_paint(frame_resized, scale_img);

                if(this->learning_central_face)
                {
                    this->learn_central_face();
                }
            }
            else
            {
                this->faces_data.clear();
            }
            this->view_image(frame_resized);
            std::cout << dff.str();

        }
        else if(frame.empty())
        {
            this->video.release();
        }

        cclock::time_point frame_t2 = cclock::now();
        std::chrono::duration<double> frame_time_dur = std::chrono::duration_cast<std::chrono::duration<double>>(frame_t2 - frame_t1);
        double file_fps = 1.0/frame_time_dur.count();
        cout << "(" << std::round(file_fps) << "fps) ";
        this->took_time("", frame_t1, frame_t2);
        frame_id++;

        QApplication::processEvents();
    }

    cclock::time_point file_t2 = cclock::now();
    this->took_time("Analysing file", file_t1, file_t2);
}


void Face_Worker::faces_track(std::vector<struct Face_Data> new_faces_data)
{
    for(auto ito = this->faces_data.begin(); ito != this->faces_data.end(); ito++)
    {
        ito->is_tracked = false;
        for(auto itn = new_faces_data.begin(); itn != new_faces_data.end(); itn++)
        {
            bool is_tracked_face_data_distance =
                    this->test_distance(itn->center, ito->center,
                                        ito->face_rect.height(), ito->face_rect.width());
            bool is_tracked_face_data_length = this->fdlib->identify_faces(ito->face.descriptor, itn->face.descriptor, this->faces_threshold);
            if(is_tracked_face_data_distance && is_tracked_face_data_length)
            {
                ito->face.descriptor = itn->face.descriptor;
                ito->face_rect = itn->face_rect;
                ito->face = itn->face;
                ito->faces.push_back(ito->face);

                ito->center = itn->center;

                ito->is_tracked = true;
                ito->tracked_frames++;

                new_faces_data.erase(itn--);
            }
        }
        if(!ito->is_tracked)
        {
            this->faces_data.erase(ito--);
        }
    }
    this->faces_data.insert(this->faces_data.end(), new_faces_data.begin(), new_faces_data.end());
}

bool Face_Worker::test_distance(dlib::point new_face_center, dlib::point old_face_center, double old_face_height, double old_face_width)
{
    bool is_distance = true;
    double distance = dlib::length(new_face_center - old_face_center);

    if(distance > 0)
    {
        dlib::line faces_line(old_face_center, new_face_center);
        dlib::line faces_hor(old_face_center, dlib::point(old_face_center.x()+1, old_face_center.y()));
        double angle = dlib::angle_between_lines(faces_hor, faces_line);

        double distance_threshold = old_face_height;
        if(angle <= 45 && angle >= -45)
        {
            distance_threshold = old_face_width;
        }
        is_distance = distance < (distance_threshold * 0.25);
    }
    return is_distance;
}

void Face_Worker::faces_identify()
{
    for (auto &face_data : this->faces_data)
    {
        double identified_face_vote = 0.0;
        (&face_data)->faces_voting.clear();

        for(auto faceID : faceIDs)
        {
            double num_face_descriptors = faceID.faces.size();
            identified_face_vote = 0;

            for(auto fid : faceID.faces)
            {
                double identified_face_tmp = this->fdlib->identify_faces_length(face_data.face.descriptor, fid.descriptor);
                if(identified_face_tmp < this->faces_threshold)
                    identified_face_vote++;
            }
            (&face_data)->faces_voting.push_back(identified_face_vote / num_face_descriptors);
        }

        std::vector<double>::iterator max;
        max = std::max_element(face_data.faces_voting.begin(), face_data.faces_voting.end());
        size_t index = size_t(std::distance(face_data.faces_voting.begin(), max));

        //std::cout << "name: " << face_data.name << " " << index;
        if(std::fabs(face_data.faces_voting.at(index)) > 0.001)
        {
            (&face_data)->is_faceID = face_data.faces_voting[index];
            (&face_data)->name = this->faceIDs[index].name;
        }
    }
}

double Face_Worker::faces_identify_get_is_faceID(Face_Data face_data, std::string name)
{
    double is_faceID = 0.0;
    if(!face_data.is_tracked)
    {
        return is_faceID;
    }

    auto itj = std::find_if(faceIDs.begin(), faceIDs.end(), boost::bind(&FaceID::name, _1) == name);

    bool identified_face = false;
    double identified_face_vote = 0.0;
    double num_face_descriptors = itj->faces.size();

    for(auto face : itj->faces)
    {
        identified_face = this->fdlib->identify_faces(face_data.face.descriptor, face.descriptor);
        if(identified_face)
            identified_face_vote++;
    }

    is_faceID = identified_face_vote / num_face_descriptors;

    return is_faceID;
}

void Face_Worker::update_faceIDs()
{
    bool face_is_new = true;
    for(auto face_data : this->faces_data)
    {
        if(face_data.is_faceID > this->faces_voting_threshold)
        {
            FaceID* faceID = this->search_faceID(face_data.name);
            if(!faceID->is_saved)
            {
                for(auto &face_data_face : face_data.faces)
                {
                    face_is_new = test_is_new(faceID->faces, face_data_face.descriptor);
                    if(face_is_new)
                    {
                        (&face_data_face)->is_saved = false;
                        faceID->faces.push_back(face_data_face);
                    }
                }
            }
        }
    }
}

void Face_Worker::faces_unknown_identify()
{
    for(auto &face_data : this->faces_data)
    {
        if(face_data.is_unknown)
        {
            //ask for name of new faceID
            if(face_data.tracked_frames >= 50)
            {
                this->faces_add_faceID(face_data);

                this->face_asked_mutex.lock();

                this->unknown_wait_until = true;

                this->face_asked_mutex.unlock();

                emit asked_new_name();
                while(this->unknown_wait_until)
                {
                    QApplication::processEvents();
                }

                this->face_asked_mutex.lock();

                std::string new_name = this->fmwin->get_unknown_new_face_name();
                if(new_name != "")
                {
                    FaceID* faceID = this->search_faceID(face_data.name);
                    if(faceID != nullptr)
                    {
                        faceID->name = new_name;
                        faceID->name = this->faces_save_faceID(faceID, false);
                        (&face_data)->name = faceID->name;
                    }
                    (&face_data)->is_unknown = false;
                }

                this->face_asked_mutex.unlock();
            }
            //add new faces to unknown faceID
            else
            {
                this->faces_add_faceID(face_data);
            }
        }
        if(!face_data.is_unknown && face_data.is_faceID <= this->faces_voting_threshold && face_data.tracked_frames >= 10)
        {
            (&face_data)->is_unknown = true;
            string new_name = this->generate_new_faceID(face_data, true);
            (&face_data)->name = new_name;
        }
    }
}

void Face_Worker::faces_add_faceID(Face_Data face_data)
{
    if(face_data.is_faceID > this->faces_voting_threshold)
    {
        FaceID* faceID = this->search_faceID(face_data.name);
        if(faceID != nullptr)
        {
            bool is_new = true;
            for(auto &face_data_face : face_data.faces)
            {
                is_new = true;
                if(!face_data_face.is_saved)
                {
                    is_new = this->test_is_new(faceID->faces, face_data_face.descriptor);
                    if(is_new)
                    {
                        faceID->faces.push_back(face_data.face);
                        (&face_data_face)->is_saved = false;
                    }
                }
            }
        }
    }
}

std::string Face_Worker::faces_save_faceID(FaceID *faceID, bool check_name)
{
    namespace bfs = boost::filesystem;

    ostringstream faceID_name;
    faceID_name << faceID->name;
    ostringstream faceID_path_name;
    faceID_path_name << this->path_to_faces_database << faceID->name;
    bfs::path path_faceID{faceID_path_name.str()};
    if(!bfs::is_directory(path_faceID))
    {
        cout << "Create save" << endl;
        bfs::create_directory(path_faceID);
    }
    else if (check_name)
    {
        size_t folder_index = 0;
        std::string folder_faceID_old = faceID_path_name.str();
        ostringstream folder_faceID;
        do
        {
            folder_faceID.str(string{});
            folder_faceID.clear();

            folder_faceID << folder_faceID_old << "_" << folder_index;

            path_faceID = folder_faceID.str();
            folder_index++;
        } while(bfs::is_directory(path_faceID));
        cout << "Create save 2" << endl;
        bfs::create_directory(path_faceID);
        faceID_name << "_" << folder_index;
    }
    ostringstream face_faceID_folder_name;
    face_faceID_folder_name << path_faceID.string() << "/" << faceID->name << "_";
    ostringstream face_faceID_file_name;
    size_t face_index = 0;
    bfs::path path_face;
    for(auto &face : faceID->faces)
    {
        if(!face.is_saved)
        {
            do
            {
                face_faceID_file_name.str(string{});

                face_faceID_file_name << face_faceID_folder_name.str() << std::setfill('0') << std::setw(4) << face_index  << ".png";

                path_face = face_faceID_file_name.str();

                face_index++;
            } while (bfs::exists(path_face));
            dlib::save_png(face.image, path_face.string());
            (&face)->is_saved = true;
        }
    }
    return faceID_name.str();
}

void Face_Worker::face_unknown_identified()
{
    this->face_asked_mutex.lock();

    this->unknown_wait_until = false;

    this->face_asked_mutex.unlock();
}

void Face_Worker::set_learn_central_face(QString central_face_new_name)
{
    face_central_mutex.lock();

    this->learning_central_face = true;
    this->central_face_new_name = central_face_new_name.toStdString();
    cout << "Central Face: " << this->central_face_new_name << endl;

    face_central_mutex.unlock();
}

void Face_Worker::learn_central_face()
{
    cout << "Set central face " << this->center_face_data->name  << endl;

    face_central_mutex.lock();

    if(this->central_face_new_name != "" && this->center_face_data != nullptr)
    {
        if(this->center_face_data->is_faceID >= this->faces_voting_threshold)
        {
            cout << "ph 1 rename" << endl;
            this->center_face_data->name = this->rename_faceID(this->central_face_new_name, this->center_face_data->name);
        }
        else
        {
            cout << "ph 1 generate" << endl;
            this->center_face_data->name = this->central_face_new_name;
            this->center_face_data->name = this->generate_new_faceID(Face_Data(*this->center_face_data), false);
        }

        cout << "ph 2 " << this->center_face_data->name << endl;
        FaceID* faceID_tmp = this->search_faceID(this->center_face_data->name);
        if(faceID_tmp != nullptr)
        {
            faceID_tmp->name = this->faces_save_faceID(faceID_tmp, false);
            this->center_face_data->name = faceID_tmp->name;
        }
        cout << "ph 3 " << this->center_face_data->name << endl;
    }
    this->learning_central_face = false;
    emit set_button_learn(true);

    face_central_mutex.unlock();
}

std::string Face_Worker::rename_faceID(std::string face_name_new, std::string face_name_old)
{
    cout << face_name_new << " - " << face_name_old << endl;
    namespace bfs = boost::filesystem;
    bfs::path faceID_old_path {this->path_to_faces_database + face_name_old};
    bfs::path faceID_new_path {this->path_to_faces_database + face_name_new};

    if(bfs::is_directory(faceID_old_path) && !bfs::is_directory(faceID_new_path))
    {
        bfs::rename(faceID_old_path, faceID_new_path);
        cout << "renamed normal: " << faceID_new_path << endl;
    }
    else
    {
        ostringstream faceID_new_path_tmp;
        size_t index = 0;

        do
        {
            index++;
            faceID_new_path_tmp.str(string{});
            faceID_new_path_tmp.clear();

            faceID_new_path_tmp << this->path_to_faces_database << face_name_new << " " << std::setfill('0') << std::setw(2) << index;

        } while(bfs::is_directory(bfs::path{faceID_new_path_tmp.str()}));
        cout << "renamed double" << endl;
        bfs::rename(faceID_old_path, bfs::path{faceID_new_path_tmp.str()});

        faceID_new_path_tmp.str(string{});
        faceID_new_path_tmp.clear();

        faceID_new_path_tmp << face_name_new << " " << std::setfill('0') << std::setw(2) << index;
        face_name_new = faceID_new_path_tmp.str();

    }

    FaceID* faceID = this->search_faceID(face_name_old);
    if(faceID != nullptr)
    {
        faceID->name = face_name_new;
    }


    return face_name_new;
}

FaceID* Face_Worker::search_faceID(std::string face_name)
{
    FaceID faceID_tmp;
    faceID_tmp.name = "";
    for(auto &faceID : this->faceIDs)
    {
        if(faceID.name == face_name)
        {
            return (&faceID);
        }
    }
    return nullptr;
}

void Face_Worker::faces_set_center(Mat frame)
{
    face_central_mutex.lock();

    //detect faces nearest to center
    dlib::point frame_center(long(std::round(frame.cols/2.0)), long(std::round(frame.rows/2.0)));
    double distance;
    double distance_min = std::max(frame.cols*2.0, frame.rows*2.0);
    size_t num_faces_data = this->faces_data.size();
    size_t index_min = num_faces_data+1;
    for(size_t i = 0; i < num_faces_data; i++)
    {
        auto &face_data_tmp = this->faces_data[i];
        face_data_tmp.is_center = false;
        distance = dlib::length(face_data_tmp.center - frame_center);
        if(distance < distance_min)
        {
            index_min = i;
            distance_min = distance;
        }
    }

    if(index_min <= num_faces_data)
    {
        this->faces_data[index_min].is_center = true;
        this->center_face_data = &(this->faces_data[index_min]);
    }

    face_central_mutex.unlock();
}

bool Face_Worker::test_is_new(vector<Face> faces, dlib::matrix<float,0,1> face_descriptor)
{
    bool is_new = true;
    for(auto faceID_face : faces)
    {
        double faces_distance = this->fdlib->identify_faces_length(face_descriptor, faceID_face.descriptor);
        if(faces_distance < 0.01)
        {
            is_new = false;
            break;
        }
    }
    return is_new;
}

cv::Mat Face_Worker::faces_paint(cv::Mat frame, double scaling)
{
    //paint rects of faces and name of identified faces
    for(auto face_data : this->faces_data)
    {
        cv::Rect cvrct = this->change_rect(face_data.face_rect);
        cvrct = this->scale_rect(cvrct, scaling);

        if(face_data.name == "unknown")
        {
            //white
            cv::rectangle(frame, cvrct, cv::Scalar(238,235,255));
        }
        else
        {
            ostringstream name_text;
            if(face_data.is_faceID > this->faces_voting_threshold)
            {
                name_text << face_data.name;
            }

            if(face_data.tracked_frames > 0)
            {
                name_text << " " << face_data.tracked_frames;
            }
            cv::Scalar color_face;

            if(face_data.is_center)
            {
                if(face_data.is_faceID > this->faces_voting_threshold)
                {
                    // dark orange
                    color_face = cv::Scalar(79,213,255);
                }
                else
                {
                    // bright orange
                    color_face = cv::Scalar(179,236,255);
                }
            }
            else if(face_data.is_faceID > this->faces_voting_threshold)
            {
                //green
                color_face = cv::Scalar(129,213,174);
            }
            else
            {
                //yellow
                color_face = cv::Scalar(118,241,255);
                name_text << " " << face_data.is_faceID;
            }

            cv::rectangle(frame, cvrct, color_face);
            cv::putText(frame, name_text.str(),cv::Point(cvrct.x, cvrct.y), cv::FONT_HERSHEY_PLAIN, 1.1, color_face, 1);
        }
    }

    return frame;
}

void Face_Worker::setup_models(std::string path_to_modelSP, std::string path_to_modelNet)
{
    this->set_status("Setup Models");

    this->fdlib->set_net(path_to_modelNet);
    this->fdlib->set_sp(path_to_modelSP);
}

void Face_Worker::faces_save_identify(size_t frame_id)
{
    namespace bfs = boost::filesystem;
    ostringstream pathSS;
    bfs::path path_file(this->path_to_file);

    bfs::path path_face_name;

    size_t fid = 0;
    ostringstream face_name;
    ostringstream face_name_index;
    string filename;
    string face_data_name;

    for(auto face_data : this->faces_data)
    {
        if(!face_data.face.is_saved)
        {
            face_data_name = face_data.name;

            path_face_name.clear();
            path_face_name = bfs::path(this->path_to_file + face_data_name);
            if(!bfs::is_directory(path_face_name))
            {
                cout << "Create save identify" << endl;
                bfs::create_directory(path_face_name);
            }

            face_data_name = std::regex_replace(face_data_name, std::regex(", "), "_");

            face_name.str(string{});
            face_name.clear();
            if(face_data.name != "unknown")
            {
                face_name << face_data.name << "/" << face_data_name << "_";
            }
            else
            {
                face_name << "unknown/unknown_";
            }

            do
            {
                face_name_index.str(string{});
                face_name_index.clear();
                fid++;

                face_name_index << frame_id << "_" << fid;

                pathSS.str(string{});
                pathSS.clear();
                pathSS << this->path_to_faces_database << face_name.str() << face_name_index.str() << ".png";
                filename = pathSS.str();
            } while(!boost::filesystem::exists(boost::filesystem::path{filename}));

            Mat cv_face = dlib::toMat(face_data.face.image);
            cv::cvtColor(cv_face,cv_face, cv::COLOR_BGR2RGB);
            cv::imwrite(filename, cv_face);
        }
    }
}

void Face_Worker::faces_save_unknown(Face_Data face_data_unknown)
{
    namespace bfs = boost::filesystem;
    ostringstream index;
    ostringstream faces_file_name;
    string face_file_name_str;
    ostringstream faces_folder;
    faces_folder << this->path_to_faces_database << "unknown/" << bfs::path(this->path_to_file).stem().string();
    bfs::path face_folder = bfs::path(faces_folder.str());
    bfs::path face_file;
    if(!bfs::is_directory(face_folder))
    {
        cout << "create unknown" << endl;
        bfs::create_directory(face_folder);
    }

    do
    {
        this->new_face_unknown_index++;
        index.str(string{});
        index.clear();
        index << std::setfill('0') << std::setw(5) << this->new_face_unknown_index;
        faces_file_name.str(string{});
        faces_file_name.clear();
        faces_file_name << faces_folder.str() << "/" << face_data_unknown.name << "_" << index.str() << ".png";
        face_file_name_str = faces_file_name.str();
        face_file = bfs::path{face_file_name_str};
    } while (bfs::exists(face_file));
    Mat cv_face = dlib::toMat(face_data_unknown.face.image);
    cv::cvtColor(cv_face,cv_face, cv::COLOR_BGR2RGB);
    cv::imwrite(face_file_name_str, cv_face);
}

std::string Face_Worker::generate_new_faceID(Face_Data face_data, bool unknown)
{
    FaceID faceID_unknown;

    ostringstream new_faceID_name;
    size_t t;
    string name_tmp;

    if(unknown)
    {
        new_faceID_name << "unknown_00";
        name_tmp = "unknown";
        t = 1;
    }
    else
    {
        new_faceID_name << face_data.name;
        name_tmp = face_data.name;
        t = 0;
    }

    bool is_known_name;
    do
    {
        is_known_name = false;
        for(auto faceID: this->faceIDs)
        {
            if(faceID.name == new_faceID_name.str())
            {
                is_known_name = true;
                break;
            }
        }
        if(is_known_name)
        {
            new_faceID_name.str(string{});
            new_faceID_name.clear();
            new_faceID_name << name_tmp << " " << std::setfill('0') << std::setw(2) << t;
            t++;
        }
    } while (is_known_name);

    faceID_unknown.name = new_faceID_name.str();
    faceID_unknown.is_saved = false;

    bool is_new = true;
    for(auto face_unknown : face_data.faces)
    {
        is_new = this->test_is_new(faceID_unknown.faces, face_unknown.descriptor);
        if(is_new)
        {
            faceID_unknown.faces.push_back(face_unknown);
        }
    }

    std::cout << "Create new faceID " << faceID_unknown << std::endl;
    this->faceIDs.push_back(faceID_unknown);
    return faceID_unknown.name;
}

void Face_Worker::change_QStatus(QString qstatus)
{
    emit status_changed(qstatus);
}

void Face_Worker::set_status(string status)
{
    emit status_changed(QString::fromStdString(status));
}


void Face_Worker::view_image(cv::Mat src)
{
    QImage qimg = QImage( reinterpret_cast<uchar*>(src.data), src.cols, src.rows, int(src.step1()), QImage::Format_RGB888);
    qimg = qimg.rgbSwapped();
    emit image_changed(qimg);
}

cv::Rect Face_Worker::change_rect(dlib::rectangle drct)
{
    return cv::Rect(int(std::round(drct.left())),int(std::round(drct.top())),int(std::round(drct.width())),int(std::round(drct.height())));
}

cv::Rect Face_Worker::scale_rect(cv::Rect rct, double scale)
{
    //cv::Size scl = new cv::Size(int(std::round(double(rct.width) * scale)), int(std::round(double(rct.height) * scale)));
    //rct += scl;
    return cv::Rect(int(std::round(rct.x * scale)), int(std::round(rct.y * scale)), int(std::round(rct.height * scale)), int(std::round(rct.width * scale)));
}

cv::Mat Face_Worker::scale_image(Mat src, double scale)
{
    Mat tmp;
    cv::resize(src,tmp, cv::Size(int(std::round(src.cols * scale)), int(std::round(src.rows * scale))), 0,0, ((scale < 1.0) ? cv::INTER_AREA : cv::INTER_LINEAR));
    return tmp;
}

cv::Mat Face_Worker::load_image(string path_to_image)
{
    this->set_status("Load Image");
    cv::Mat cvimg;
    cvimg = cv::imread(path_to_image, CV_LOAD_IMAGE_COLOR);
    return cvimg;
}

void Face_Worker::took_time(string title, std::chrono::high_resolution_clock::time_point timestamp1, std::chrono::high_resolution_clock::time_point timestamp2)
{
    std::chrono::duration<double> time_dur = std::chrono::duration_cast<std::chrono::duration<double>>(timestamp2 - timestamp1);
    ostringstream title_time;
    title_time << title << " took " << std::setprecision(2) << time_dur.count() << "s\n";
    std::cout << title_time.str() << "\n";
}

void Face_Worker::set_view_size(cv::Size size)
{
    this->viewSize = size;
}

void Face_Worker::close_window()
{
    cout << "EXIT" << endl;
    emit window_closed();
}
