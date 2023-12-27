#pragma once

#include "behaviortree_cpp_v3/bt_factory.h"

using namespace BT;

namespace myTree
{
    class At : public BT::ConditionNode
    {
        public:
            At(const std::string& name, const BT::NodeConfiguration& config)
            : BT::ConditionNode(name, config)
            {
            }

            static BT::PortsList providedPorts()
            {
                return{ InputPort<std::string>("param")};
            }

            BT::NodeStatus tick() override;

    }; 

    class Move1 : public BT::SyncActionNode
    {
        public:
            Move1(const std::string& name, const BT::NodeConfiguration& config)
            : BT::SyncActionNode(name, config)
            {
            }

            static BT::PortsList providedPorts()
            {
                return{ InputPort<std::string>("param")};
            }

            BT::NodeStatus tick() override;

    }; 

    class Move2 : public BT::SyncActionNode
    {
        public:
            Move2(const std::string& name, const BT::NodeConfiguration& config)
            : BT::SyncActionNode(name, config)
            {
            }

            static BT::PortsList providedPorts()
            {
                return{ InputPort<std::string>("param")};
            }

            BT::NodeStatus tick() override;

    }; 

    BT::NodeStatus Open();

    BT::NodeStatus OpenPump();

    BT::NodeStatus Close();

    BT::NodeStatus ClosePump();

    BT::NodeStatus PumpReady();

    class Have : public BT::ConditionNode
    {
        public:
            Have(const std::string& name, const BT::NodeConfiguration& config)
            : BT::ConditionNode(name, config)
            {
            }

            static BT::PortsList providedPorts()
            {
                return{ InputPort<std::string>("param")};
            }

            BT::NodeStatus tick() override;

    };
    
    void RegisterNodes(BT::BehaviorTreeFactory& factory);

}

