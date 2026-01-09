import { Button, Divider } from "@fluentui/react-components";
import { ChatAdd24Regular } from "@fluentui/react-icons";
import "./NavBar.css";

interface Props {
    onNewChat: () => void;
}

export const NavBar = ({ onNewChat }: Props) => {
    return (
        <div className="sidebar-content">
            <div className="sidebar-section">
                <Button appearance="primary" icon={<ChatAdd24Regular />} className="new-chat-btn" onClick={onNewChat} size="large">
                    New Chat
                </Button>
            </div>

            <Divider className="sidebar-divider" />

            <div className="sidebar-section scrollable"></div>
        </div>
    );
};
