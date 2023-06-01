//
// Created by Dongho Kang on 10.09.22.
//

#ifndef PYLOCO_VANILLASIMULATOR_H
#define PYLOCO_VANILLASIMULATOR_H

#include "crl-basic/gui/plots.h"
#include "pylocobase/sim/Simulator.h"
#include "loco/simulation/ode/ODERBEngine.h"

namespace pyloco {

/**
 * Simulator for Vanilla policy which generates joint-level P targets from user's high-level command.
 */
class VanillaSimulator : public Simulator {
public:
    double commandForwardSpeed = 1.0;

private:
    // plots
    std::shared_ptr<crl::gui::RealTimeLinePlot2D<crl::dVector>> baseSpeedPlot = nullptr;
    std::shared_ptr<crl::gui::RealTimeLinePlot2D<crl::dVector>> jointAnglePlot = nullptr;
    std::shared_ptr<crl::gui::RealTimeLinePlot2D<crl::dVector>> jointSpeedPlot = nullptr;
    std::shared_ptr<crl::gui::RealTimeLinePlot2D<crl::dVector>> jointAccelerationPlot = nullptr;
    std::shared_ptr<crl::gui::RealTimeLinePlot2D<crl::dVector>> jointTorquePlot = nullptr;
    std::shared_ptr<crl::gui::RealTimeLinePlot2D<crl::dVector>> jointActionPlot = nullptr;
    std::shared_ptr<crl::gui::RealTimeLinePlot2D<crl::dVector>> outputTorquePlot = nullptr;


public:
    explicit VanillaSimulator(double simTimeStepSize,
                     double controlTimeStepSize,
                     uint robotModel = (uint)RobotInfo::Model::Dog, bool loadVisuals = false) : Simulator(simTimeStepSize, controlTimeStepSize, robotModel, loadVisuals) {
        baseSpeedPlot = std::make_shared<crl::gui::RealTimeLinePlot2D<crl::dVector>>("Base Velocity", "[sec]", "[m/s]");
        jointAnglePlot = std::make_shared<crl::gui::RealTimeLinePlot2D<crl::dVector>>("Joint Angle", "[sec]", "[rad]");
        jointSpeedPlot = std::make_shared<crl::gui::RealTimeLinePlot2D<crl::dVector>>("Joint Speed", "[sec]", "[rad/s]");
        jointAccelerationPlot = std::make_shared<crl::gui::RealTimeLinePlot2D<crl::dVector>>("Joint Acceleration", "[sec]", "[rad/s^2]");
        jointTorquePlot = std::make_shared<crl::gui::RealTimeLinePlot2D<crl::dVector>>("Joint Torque", "[sec]", "[N.m]");
        jointActionPlot = std::make_shared<crl::gui::RealTimeLinePlot2D<crl::dVector>>("Joint Action Angle", "[sec]", "[rad]");
        outputTorquePlot = std::make_shared<crl::gui::RealTimeLinePlot2D<crl::dVector>>("Output Joint Torque", "[sec]", "[N.m]");

        for (int i = 0; i < robot_->getJointCount(); i++) {
            jointAnglePlot->addLineSpec({robot_->getJoint(i)->name, [i](const auto &d) { return (float)d[i]; }});
            jointSpeedPlot->addLineSpec({robot_->getJoint(i)->name, [i](const auto &d) { return (float)d[i]; }});
            jointAccelerationPlot->addLineSpec({robot_->getJoint(i)->name, [i](const auto &d) { return (float)d[i]; }});
            jointTorquePlot->addLineSpec({robot_->getJoint(i)->name, [i](const auto &d) { return (float)d[i]; }});
            jointActionPlot->addLineSpec({robot_->getJoint(i)->name, [i](const auto &d) { return (float)d[i]; }});
            outputTorquePlot->addLineSpec({robot_->getJoint(i)->name, [i](const auto &d) { return (float)d[i]; }});
        }
        baseSpeedPlot->addLineSpec({"Base Velocity", [](const auto &d) { return (float)d[0]; }});
        baseSpeedPlot->addLineSpec({"Target Velocity", [](const auto &d) { return (float)d[1]; }});
        allLoopMotorTorques.resize(robot_->getJointCount() * int(controlTimeStepSize / simTimeStepSize + 1.0e-10));
    }

    ~VanillaSimulator() override = default;

    void reset() override {
        Simulator::reset();
        robot_->setRootState(crl::P3D(0, nominalBaseHeight_, 0), crl::Quaternion::Identity());
        baseSpeedPlot->clearData();
        jointAnglePlot->clearData();
        jointSpeedPlot->clearData();
        jointAccelerationPlot->clearData();
        jointTorquePlot->clearData();
        jointActionPlot->clearData();
        outputTorquePlot->clearData();
        allLoopMotorTorques.setZero();
    }

    template <size_t N>
    bool contains(int (&arr)[N], int itemToFind) {
        for (int i: arr){
            if (i == itemToFind){
                return true;
            }
        }
        return false;
    }

    void step(const crl::dVector &jointTarget, float curr_max_episode_length) override {
        crl::dVector initial_joint_angles = getQ().tail(numJoints_);
        allLoopMotorTorques.setZero();  // set to zero

        robot_->root->rbProps.mass = 6.4072265625;
        for (int i = 0; i < robot_->jointList.size(); i++) {
            int a[] = {0,1,2,3,4,5,9,12,13,14,18,19,20,23,27,28,29,30,31,32,40,41};
            if (contains(a, i)) {
                robot_->jointList[i]->child->rbProps.mass = 0.35;
            }

            int b[] = {6,7,8};
            if (contains(b, i)) {
                robot_->jointList[i]->child->rbProps.mass = 5.6953125;
            }

            int c[] = {24,25,26};
            if (contains(c, i)) {
                robot_->jointList[i]->child->rbProps.mass = 0.35595703125;
            }

            int d[] = {36,37,38,39};
            if (contains(d, i)) {
                robot_->jointList[i]->child->rbProps.mass = 0.8009033203125;
            }

            int e[] = {36,37,38,39};
            if (contains(e, i)) {
                robot_->jointList[i]->child->rbProps.mass = 0.8009033203125;
            }

            if (i == 42 || i == 43) {
                robot_->jointList[i]->child->rbProps.mass = 0.40045166015625;
            }

            if (i == 33 || i == 34) {
                robot_->jointList[i]->child->rbProps.mass = 2.4027099609375;
            }

            if (i == 21 || i == 22) {
                robot_->jointList[i]->child->rbProps.mass = 0.13348388671875;
            }

            if (i == 16 || i == 17) {
                robot_->jointList[i]->child->rbProps.mass = 0.533935546875;
            }

            if (i == 10 || i == 11) {
                robot_->jointList[i]->child->rbProps.mass = 3.20361328125;
            }

            if (i == 15) {
                robot_->jointList[i]->child->rbProps.mass = 19.2216796875;
            }

            if (i == 35) {
                robot_->jointList[i]->child->rbProps.mass = 4.271484375;
            }
        }

        if (curr_max_episode_length < 40) {
            robot_->root->rbProps.mass = 0.01;
            for (uint i = 0; i < robot_->jointList.size(); i++) {
                robot_->jointList[i]->child->rbProps.mass = 0.01;
            }
        } else if (curr_max_episode_length >= 40 && curr_max_episode_length < 50) {
            robot_->root->rbProps.mass *= 0.25;
            for (uint i = 0; i < robot_->jointList.size(); i++) {
                robot_->jointList[i]->child->rbProps.mass *= 0.25;
            }
        } else if (curr_max_episode_length >= 50 && curr_max_episode_length < 60) {
            robot_->root->rbProps.mass *= 0.5;
            for (uint i = 0; i < robot_->jointList.size(); i++) {
                robot_->jointList[i]->child->rbProps.mass *= 0.5;
            }
        } else if (curr_max_episode_length >= 60 && curr_max_episode_length < 70) {
            robot_->root->rbProps.mass *= 0.75;
            for (uint i = 0; i < robot_->jointList.size(); i++) {
                robot_->jointList[i]->child->rbProps.mass *= 0.75;
            }
        } else if (curr_max_episode_length >= 70) {
            robot_->root->rbProps.mass *= 1;
            for (uint i = 0; i < robot_->jointList.size(); i++) {
                robot_->jointList[i]->child->rbProps.mass *= 1;
            }
        }

        applyControlSignal(jointTarget);
        jointActionPlot->addData((float)getTimeStamp(), jointTarget);


        double simTime = 0;
        int numSimStepsPerLoop = (int)((controlTimeStepSize + 1e-10) / simTimeStepSize);

        for (int i = 0; i < numSimStepsPerLoop; i++) {
            crl::dVector PastJointSpeed = getQDot().tail(numJoints_);

            // step the simulation
            simulationStep();

            allLoopMotorTorques.segment(i * numJoints_, numJoints_) = getMotorTorques();

            // populate plots
            jointAnglePlot->addData((float)getTimeStamp(), getQ().tail(numJoints_));
            jointSpeedPlot->addData((float)getTimeStamp(), getQDot().tail(numJoints_));
            jointTorquePlot->addData((float)getTimeStamp(), getMotorTorques());
            crl::dVector JointAcceleration = (getQDot().tail(numJoints_) - PastJointSpeed) / simTimeStepSize;
            jointAccelerationPlot->addData((float)getTimeStamp(), JointAcceleration);
            crl::dVector baseSpeeds(2);
            baseSpeeds[0] = robot_->getRoot()->getLocalCoordinates(robot_->getRoot()->getVelocityForPoint_local(crl::P3D())).z();
            baseSpeeds[1] = commandForwardSpeed;
            baseSpeedPlot->addData((float)getTimeStamp(), baseSpeeds);

            // advance in time
            simTime += simTimeStepSize;
        }
        //        outputTorquePlot->addData((float)getTimeStamp(), getMotorTorques());
    }

    void drawImPlot() const override {
        jointAnglePlot->draw();
        jointActionPlot->draw();
        jointSpeedPlot->draw();
        jointAccelerationPlot->draw();
        jointTorquePlot->draw();
        baseSpeedPlot->draw();
        //        outputTorquePlot->draw();
    }

    crl::dVector getAllLoopMotorTorques() const {
        return allLoopMotorTorques;
    }
};

}  // namespace pyloco

#endif  //PYLOCO_VANILLASIMULATOR_H
