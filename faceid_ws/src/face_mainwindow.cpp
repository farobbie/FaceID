#include "face_mainwindow.h"
#include "ui_face_mainwindow.h"

#include <opencv2/opencv.hpp>
#include <iostream>

Face_Mainwindow::Face_Mainwindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Face_Mainwindow)
{
    ui->setupUi(this);

    this->setWindowTitle("FaceID");

    this->status_label = new QLabel(this);
    this->status_runtime = new QLabel(this);
    this->status_runtime->setAlignment(Qt::AlignRight);
    ui->status_bar->addPermanentWidget(this->status_label);
    ui->status_bar->addPermanentWidget(this->status_runtime);
    ui->status_bar->setVisible(true);


    this->face_scene = new QGraphicsScene(this);
    ui->view_frame->setScene(this->face_scene);
    ui->view_frame->scene()->addItem(&(this->face_pixmap));
    ui->view_frame->show();

    connect(this->ui->button_learn, &QPushButton::clicked, this, &Face_Mainwindow::on_button_learn_clicked);
}

Face_Mainwindow::~Face_Mainwindow()
{
    delete ui;
}

void Face_Mainwindow::closeEvent(QCloseEvent *event)
{
    emit window_closed();
    event->accept();
    //QMainWindow::closeEvent(event);
}

void Face_Mainwindow::change_status(QString status)
{
    status.insert(0, "Status: ");
    this->ui->status_bar->showMessage(status);
    this->repaint();
}

void Face_Mainwindow::change_image(QImage qimg)
{
    this->face_pixmap.setPixmap(QPixmap::fromImage(qimg));
}


cv::Size Face_Mainwindow::get_view_size()
{
    return cv::Size(this->ui->view_frame->width(), this->ui->view_frame->height());
}

void Face_Mainwindow::ask_for_new_name()
{
    bool ok;
    QString face_name = QInputDialog::getText(this, tr("New Face Identified"), tr("name of face:"), QLineEdit::Normal, tr(""), &ok);
    this->unknown_new_face_name = "";
    if (ok && !face_name.isEmpty())
    {
        this->unknown_new_face_name = face_name.toStdString();
    }
    emit face_identified();
}

std::string Face_Mainwindow::get_unknown_new_face_name()
{
    return this->unknown_new_face_name;
}

void Face_Mainwindow::show()
{
    QMainWindow::show();
    QApplication::processEvents();

    emit window_shown();
    if(this->first_time_shown)
    {
        emit window_loaded();
        this->first_time_shown = false;
    }
}

void Face_Mainwindow::update_clock(QString qtime)
{
    this->status_runtime->setText(qtime);
}

void Face_Mainwindow::on_button_learn_clicked()
{
    if(this->ui->button_learn->isEnabled())
    {
        std::cout << "Clicked learned button" << std::endl;
        this->ui->button_learn->setEnabled(false);
        QString central_face_new_name = this->ui->face_name->text();
        emit learn_central_face(central_face_new_name);
    }
}

void Face_Mainwindow::set_button_learn(bool active)
{
    this->ui->button_learn->setEnabled(active);
}
