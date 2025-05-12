#include "face_manager.h"

#include <iostream>

namespace bfs = boost::filesystem;

Face_Manager::Face_Manager(QObject *parent) :
    QObject(parent)
{

}

Face_Manager::~Face_Manager()
{
}

void Face_Manager::setup(std::string path_to_file, std::string path_to_faces_database, std::string path_to_model_net, std::string path_to_modelSP,
                         bool save_faces, double faces_threshold, double faces_voting_threshold)
{
    this->thread_worker = new QThread();
    this->face_worker.moveToThread(this->thread_worker);

    this->thread_runtime = new QThread();
    this->moveToThread(this->thread_runtime);

    this->fdlib.set_net(path_to_model_net);
    this->fdlib.set_sp(path_to_modelSP);

    bool identify_faces = (path_to_faces_database != "");
    this->face_worker.setup_vars(path_to_file, path_to_faces_database, path_to_model_net, path_to_modelSP,
                                 save_faces, faces_threshold, faces_voting_threshold, identify_faces);

    this->face_worker.setup(&(this->fdlib), &(this->fmwin));


    connect(this->thread_worker, &QThread::started, &(this->face_worker), &Face_Worker::process);

    connect(&(this->face_worker), &Face_Worker::finished, this->thread_worker, &QThread::quit);
    connect(&(this->face_worker), &Face_Worker::finished, &(this->face_worker), &Face_Worker::deleteLater);
    connect(this->thread_worker, &QThread::finished, this->thread_worker, &QThread::deleteLater);

    connect(&(this->fmwin), &Face_Mainwindow::window_closed, &(this->face_worker), &Face_Worker::close_window);
    connect(&(this->face_worker), &Face_Worker::window_closed, this->thread_worker, &QThread::quit);
    connect(&(this->face_worker), &Face_Worker::window_closed, &(this->face_worker), &Face_Worker::deleteLater);
    connect(&(this->face_worker), &Face_Worker::window_closed, this->thread_runtime, &QThread::quit);

    connect(this->thread_runtime, &QThread::finished, this->thread_runtime, &QThread::quit);
    connect(this->thread_runtime, &QThread::finished, this->thread_runtime, &QThread::deleteLater);

    connect(&(this->face_worker), &Face_Worker::image_changed, &(this->fmwin), &Face_Mainwindow::change_image);
    connect(&(this->face_worker), &Face_Worker::status_changed, &(this->fmwin), &Face_Mainwindow::change_status);

    connect(&(this->fmwin), &Face_Mainwindow::learn_central_face, &(this->face_worker), &Face_Worker::set_learn_central_face);
    connect(&(this->face_worker), &Face_Worker::set_button_learn, &(this->fmwin), &Face_Mainwindow::set_button_learn);
    connect(&(this->face_worker), &Face_Worker::asked_new_name, &(this->fmwin), &Face_Mainwindow::ask_for_new_name);

    connect(&(this->fmwin), &Face_Mainwindow::window_closed, this, &Face_Manager::close_application);

    connect(this->thread_runtime, &QThread::started, this, &Face_Manager::start_clock);

    this->face_worker.set_view_size(this->fmwin.get_view_size());

    bfs::path unknown_database_path(path_to_faces_database.append("unknown"));
    if(!bfs::exists(unknown_database_path))
    {
        bfs::create_directory(unknown_database_path);
    }

}

void Face_Manager::start_clock()
{
    this->timer_runtime = new QTimer(this);
    this->timer_runtime->setInterval(1000);

    connect(this->timer_runtime, &QTimer::timeout, this, &Face_Manager::update_clock);
    connect(this, &Face_Manager::clock_updated, &(this->fmwin), &Face_Mainwindow::update_clock);


    this->timer_eruntime.start();
    this->timer_runtime->start();
}

void Face_Manager::update_clock()
{
    QTime tm1(0,0,0);
    tm1 = tm1.addMSecs(int(this->timer_eruntime.elapsed()));
    emit clock_updated(tm1.toString("mm:ss"));
}

void Face_Manager::start_process()
{
    this->fmwin.show();
    this->thread_runtime->start();
    this->thread_worker->start();
}

void Face_Manager::load_database(std::string path_to_database)
{
    size_t faces_index = 0;
    size_t faces_added = 0;

    std::stringstream faces_index_str;
    std::string faces_name;
    std::string faces_name_str;
    std::string filename;
    //std::stringstream new_filename;
    //std::string new_filename_str;

    bfs::path database_path(path_to_database);
    bool is_new_face;
    size_t index_name;
    size_t num_faceIDs;

    if(bfs::is_directory(database_path))
    {
        for(auto& faces_folder : boost::make_iterator_range(bfs::directory_iterator(database_path)))
        {
            if(bfs::is_directory(faces_folder))
            {
                faces_index = 0;
                faces_added = 0;
                faces_name = faces_folder.path().filename().stem().string();
                faces_name_str = std::regex_replace(faces_name, std::regex(", "), "_");
                num_faceIDs = this->faceIDs.size();

                if(faces_name != "unknown")
                {
                    bfs::path pd( string(faces_folder.path().string()).append("/double/"));

                    if(!bfs::exists(pd))
                    {
                        bfs::create_directory(pd);
                    }

                    for(index_name = 0; index_name < this->faceIDs.size(); ++index_name)
                    {
                        if(this->faceIDs[index_name].name == faces_name)
                        {
                            break;
                        }
                    }

                    if(index_name >= num_faceIDs)
                    {
                        this->faceIDs.push_back(FaceID {faces_name, std::vector<struct Face>{}});
                    }

                    FaceID faceID = this->faceIDs.at(index_name);
                    for(auto& face_file : boost::make_iterator_range(bfs::directory_iterator(faces_folder)))
                    {
                        if(bfs::is_regular_file(face_file))
                        {
                            faces_index++;

                            filename = face_file.path().string();
                            is_new_face = true;

                            matrix<rgb_pixel> face_img;
                            dlib::load_image(face_img, filename);
                            dlib::matrix<float,0,1> face_descriptor = this->fdlib.get_face_descriptor_of_face(face_img);
                            Face face_new {face_img, face_descriptor, true};

                            for(auto faceID_face : faceID.faces)
                            {
                                double is_new = this->compare_faces(face_new.descriptor, faceID_face.descriptor);
                                if(is_new < 0.001)
                                {
                                    is_new_face = false;
                                }
                            }

                            if(is_new_face)
                            {
                                //add face to faceID database
                                is_new_face = this->generate_faceID(faces_name, face_new);
                                faces_added++;
                            }
                            else
                            {
                                //move face to double-folder
                                bfs::path tmp (string(pd.string()).append(face_file.path().filename().string()));
                                bfs::rename(face_file.path(), tmp);
                            }
                        }
                    }
                    std::cout << "Added " << faces_added << " of " << faces_index << " faces to " << faces_name << std::endl;
                }
            }
        }
        this->face_worker.set_faceIDs(this->faceIDs);

        std::cout << "Database:\n";
        for(auto faceID : this->faceIDs)
        {
            std::cout << faceID << " ";
        }
        std::cout << std::endl;
    }
}

double Face_Manager::compare_faces(dlib::matrix<float,0,1> face_descriptor, dlib::matrix<float,0,1> face_descriptor_cmp)
{

    return this->fdlib.identify_faces_length(face_descriptor, face_descriptor_cmp);
}

bool Face_Manager::generate_faceID(std::string name, struct Face face)
{
    std::vector<struct Face> tmp;
    tmp.push_back(face);
    return this->generate_faceIDs(name, tmp);
}

bool Face_Manager::generate_faceIDs(std::string name, std::vector<struct Face> faces)
{
    bool is_new = true;
    size_t num_faceIDs = this->faceIDs.size();
    size_t is_known_faceID_index = num_faceIDs+1;

    for(size_t i = 0; i < num_faceIDs; ++i)
    {
        if(this->faceIDs[i].name == name)
        {
            is_known_faceID_index = i;
            break;
        }
    }

    if(is_known_faceID_index < num_faceIDs)
    {
        for(size_t n = 0; n < faces.size(); ++n)
        {
            Face new_face = faces[n];

            for(size_t o = 0; o < this->faceIDs[is_known_faceID_index].faces.size(); ++o)
            {
                Face face = this->faceIDs[is_known_faceID_index].faces[o];

                float distance = dlib::length(new_face.descriptor - face.descriptor);
                if(distance < float(0.01))
                {
                   is_new = false;
                   break;
                }
            }
            if(is_new)
            {
                this->faceIDs[is_known_faceID_index].faces.push_back(new_face);
            }
        }
    }
    else
    {
        struct FaceID faceID;
        faceID.name = name;
        faceID.faces = faces;
        this->faceIDs.push_back(faceID);
    }
    return is_new;
}


void Face_Manager::close_application()
{
    this->timer_runtime->stop();
}
