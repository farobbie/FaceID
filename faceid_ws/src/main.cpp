#include "face_manager.h"
#include "face_mainwindow.h"

#include <iostream>
#include <algorithm>
#include <iterator>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <QThread>
#include <QApplication>

namespace bpo = boost::program_options;
namespace bfs = boost::filesystem;
using cclock = std::chrono::high_resolution_clock;

template<class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(os, " "));
    return os;
}

[[noreturn]] void show_help(const bpo::options_description desc)
{
    std::cout << desc << std::endl;
    exit (EXIT_SUCCESS);
}

bool check_file(bfs::path p)
{
    try
    {
        if(bfs::exists(p))
        {
            if(bfs::is_regular_file(p))
            {
                return true;

            }
            else
            {
                std::cout << p << " isn't correct" << std::endl;
                return false;
            }
        }
        else
        {
            std::cout << p << " does not exist" << std::endl;
            return false;
        }
    }
    catch (const bfs::filesystem_error& ex)
    {
        std::cout << ex.what() << std::endl;
        return false;
    }
}

bool check_folder(bfs::path p)
{
    try
    {
        if(bfs::exists(p))
        {
            if(bfs::is_directory(p))
            {
                return true;

            }
            else
            {
                std::cout << p << " isn't correct" << std::endl;
                return false;
            }
        }
        else
        {
            std::cout << p << " does not exist" << std::endl;
            return false;
        }
    }
    catch (const bfs::filesystem_error& ex)
    {
        std::cout << ex.what() << std::endl;
        return false;
    }
}

int main(int argc, char **argv)
{
    try {
        std::cout << std::unitbuf;

        bpo::options_description desc("Usage");
        string path_to_file = "../data/faces_examples_dlib/faces/bald_guys.jpg";
        string path_to_model_net = "../data/model_dlib/dlib_face_recognition_resnet_model_v1.dat";
        string path_to_model_sp = "../data/model_dlib/shape_predictor_5_face_landmarks.dat";
        string path_to_faces_database = "";
        double faces_threshold = 0.6;
        double faces_voting_threshold = 0.5;
        bool save_faces = false;
        desc.add_options()
                ("help", "help")
                ("file", bpo::value<string>()->default_value(path_to_file) ,"path to file")
                ("faces_database", bpo::value<string>()->default_value(path_to_faces_database), "path to database of faces")
                ("save_faces", "save new identified faces in database?")
                ("modelNet", bpo::value<string>()->default_value(path_to_model_net),"path to the file of DNN responsible for face identification [download example from http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2]")
                ("modelSP", bpo::value<string>()->default_value(path_to_model_sp),"path to the file with a face landmarking model [download example from http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2]")
                ("faces_thres", bpo::value<double>()->default_value(faces_threshold),"threshold for distance of two faces")
                ("faces_voting_thres", bpo::value<double>()->default_value(faces_voting_threshold), "threshold for faces voting")
                ;

        bpo::positional_options_description p;

        bpo::variables_map vm;
        bpo::store(bpo::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
        bpo::notify(vm);

        if(vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }

        if (vm.count("file"))
        {
            path_to_file = vm["file"].as<string>();
            bfs::path pTF {path_to_file};

            if(!bfs::exists(pTF))
            {
                std::cout << "File does not exists!" << std::endl;
                return 1;
            }

            std::cout << "Using file: " << path_to_file << std::endl;
        }
        if (vm.count("faces_database"))
        {
            path_to_faces_database = vm["faces_database"].as<string>();

            if(!check_folder(path_to_faces_database))
            {
                std::cout << "Wrong database folder!" << std::endl;
                return 1;
            }

            path_to_faces_database = (std::strncmp(&path_to_faces_database.back(),"/",1) != 0) ?
                        path_to_faces_database.append("/") : (path_to_faces_database);

            std::cout << "Using faces_database " << path_to_faces_database << std::endl;
        }
        if (vm.count("save_faces"))
        {
            save_faces = true;

            std::cout << "Save new faces" << std::endl;
        }
        if (vm.count("modelNet"))
        {
            path_to_model_net = vm["modelNet"].as<string>();
            bfs::path pTN {path_to_model_net};
            if(!check_file(pTN))
            {
                return 1;
            }

            std::cout << "Using ModelNet: " << path_to_model_net << std::endl;
        }
        if (vm.count("modelSP"))
        {
            path_to_model_sp = vm["modelSP"].as<string>();
            bfs::path pTS {path_to_model_sp};
            if(!check_file(pTS))
            {
                return 1;
            }
            std::cout << "Using ModelShapePredictor: " << path_to_model_sp << std::endl;
        }
        if (vm.count("faces_thres"))
        {
            faces_threshold = vm["faces_thres"].as<double>();
            if(faces_threshold > 1.0 || faces_threshold < 0.0)
            {
                std::cout << "Wrong faces_threshold (" << faces_threshold << ")" << std::endl;
                return 1;
            }
            std::cout << "Using threshold for distance of two faces: " << faces_threshold << std::endl;
        }
        if (vm.count("faces_voting_thres"))
        {
            faces_voting_threshold = vm["faces_voting_thres"].as<double>();
            if(faces_voting_threshold > 1.0 || faces_voting_threshold < 0.0)
            {
                std::cout << "Wrong faces_voting_threshold (" << faces_voting_threshold << ")" << std::endl;
            }
            std::cout << "Using threshold for voting: " << faces_voting_threshold << std::endl;
        }


        cclock::time_point app_t1 = cclock::now();
        QApplication app(argc, argv);
        Face_Manager fmr;
        fmr.setup(path_to_file, path_to_faces_database, path_to_model_net, path_to_model_sp, save_faces, faces_threshold, faces_voting_threshold);
        if(path_to_faces_database != "")
        {
            fmr.load_database(path_to_faces_database);
        }
        fmr.start_process();
        static int app_exec = app.exec();
        cclock::time_point app_t2 = cclock::now();
        std::chrono::duration<double> app_dur = std::chrono::duration_cast<std::chrono::duration<double>>(app_t2 - app_t1);
        std::stringstream fr_time;
        std::cout << "Total time: " << app_dur.count() << "s" << std::endl;
        return app_exec;

    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;
        return 1;
    }
}
