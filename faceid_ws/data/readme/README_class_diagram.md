@startuml

interface Face{
  +image: dlib::matrix<dlib::rgb_pixel>
  +descriptor: dlib::matrix<float,0,1>
}

interface FaceID{
  +name: string
  +faces: vector<Face>
}

interface Face_Data{
  +name: string
  +face_rect: dlib::rectangle
  +face: Face
  +faces: vector<Face>
  +center: dlib::point
  +tracked_frames: size_t
  +is_faceID: double
  +is_tracked: bool = false
  +is_center: bool = false
}

class Face_Manager{
  -fmwin: Face_Mainwindow
  -fdlib: Face_DLIB
  -face_worker: Face_Worker
  -faceIDs: vector<FaceID>
  +start_process()
  +load_database(string path_to_faces_database)
  -generate_faceID(string name, Face face): bool 
}

class Face_Worker{
  -fdlib: *Face_DLIB
  -fmwin: *Face_Mainwindow
  -faces: vector<dlib::matrix<dlib::rgb_pixel>>
  -faceIDs: vector<FaceID>
  -faces_data: vector<Face_Data>
  -central_face_new_name: string
  __
  +setup_vars(...)
  +set_faceIDs(std::vector<FaceID> faceIDs)
  
  -start_analysis()
  -setup_models(string path_to_modelSP, string path_to_modelNet)
  -process_video()
  -faces_save_identify()
  -faces_save_unknown(Face_Data faces_data_unknown)
  -faces_track(vector<Face_Data> new_faces_data)
  -faces_identify()
  -faces_unknown_identify()
  -faces_add_faceID(Face_Data face_data)
  -faces_save_faceID(FaceID* faceID, bool check_name): string
  -update_faceIDs()
  
  -generate_new_faceID(Face_Data face_data, bool unknown): string
  -add_new_faces_to_unknown_faceID(vector<Face>)
  -rename_faceID(string face_name_new, string face_name_old)
  -learn_central_face()
  
  __ signals __
  asked_new_name()
  finished()
  -- slots --
  +process()
  +set_learn_central_face(QString central_face_new_name)
  +face_unknown_identified()
}

class Face_DLIB{
  -net: dlib::anet_type
  -sp: dlib::shape_predictor
  --
  +get_face_rects(matrix<rgb_pixel> img): vector<rectangle>
  +detect_faces
  (matrix<rgb_pixel> img, 
  vector<rectangle> face_rects): vector<matrix<rgb_pixel>>
  +identify_faces
  (matrix<float,0,1> face_descriptor_src, 
  matrix<float,0,1> face_descriptor_cmp, 
  double faces_threshold=0.6): bool
  +set_net(string path_to_modelNet)
  +set_sp(string path_to_modelSp)
}

class Face_Mainwindow{
  +get_unknown_ew_face_name(): string
  +ask_for_new_name()
  +set_button_learn(bool active)  
  --signals--
  learn_central_face(QString central_face_new_name)
  --slots--
  -on_button_learn_clicked()
}

Face_Worker -left-o Face_DLIB: ""
Face_Worker -left- Face_Mainwindow: ""
Face_Manager -down- Face_Worker: ""
Face_Manager -down- Face_Mainwindow: ""

Face_Mainwindow --o Face_Worker: "learn central"

Face_Data "1" *-right- "0..*" Face: ""
FaceID    "1" *-left- "0..*" Face: ""


Face_Worker "1" *-up- "0..*" Face_Data: ""
Face_Worker "1" *-up- "0..*" FaceID: ""

@enduml