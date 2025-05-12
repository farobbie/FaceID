#ifndef FACE_MAINWINDOW_H
#define FACE_MAINWINDOW_H


#include <opencv2/core/core.hpp>


#include <QInputDialog>

#include <QMainWindow>
#include <QCloseEvent>
#include <QString>
#include <QLabel>
#include <QFileDialog>
#include <QGraphicsScene>
#include <QPixmap>
#include <QGraphicsPixmapItem>
#include <QTimer>
#include <string>
#include <tuple>


namespace Ui {
class Face_Mainwindow;
}

class Face_Mainwindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit Face_Mainwindow(QWidget *parent = Q_NULLPTR);
    ~Face_Mainwindow();
    cv::Size get_view_size();
    void show();
    std::string get_unknown_new_face_name();
    bool wait_until_asked_face_name = false;

private:
    bool first_time_shown = true;
    Ui::Face_Mainwindow *ui;
    QLabel *status_label;
    QLabel *status_runtime;
    QGraphicsScene *face_scene;
    QGraphicsPixmapItem face_pixmap;
    std::string unknown_new_face_name;

public slots:
    void change_status(QString status);
    void change_image(QImage src);
    void update_clock(QString qtime);
    void ask_for_new_name();
    void set_button_learn(bool active);

protected:
    void closeEvent(QCloseEvent *event);

private slots:
    void on_button_learn_clicked();

private slots:

signals:
    void file_loaded(QString file_path);
    void window_shown();
    void window_loaded();
    void face_identified();
    void learn_central_face(QString central_face_new_name);
    void window_closed();

};

#endif // FACE_MAINWINDOW_H
