#!/bin/bash
# LangManus Launcher Script

# Make script executable
# chmod +x launch.sh

# Function to display the menu
show_menu() {
    clear
    echo "===================================="
    echo "LangManus AI System Launcher"
    echo "===================================="
    echo
    echo "Please select an option:"
    echo
    echo "1. Launch LangManus"
    echo "2. Run diagnostics"
    echo "3. Run LM Studio compatibility tests"
    echo "4. View recent feedback"
    echo "5. Exit"
    echo
}

# Function to return to menu
return_to_menu() {
    echo
    echo "Press Enter to return to the menu..."
    read
    show_menu
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice (1-5): " choice
    
    case $choice in
        1)
            clear
            echo "Launching LangManus..."
            echo
            python main.py
            return_to_menu
            ;;
        2)
            clear
            echo "Running system diagnostics..."
            echo
            python main.py --diagnostics
            return_to_menu
            ;;
        3)
            clear
            echo "Running LM Studio compatibility tests..."
            echo
            python test_lm_studio.py
            return_to_menu
            ;;
        4)
            clear
            echo "Viewing recent feedback..."
            echo
            python -c "from src.integration import feedback; recent = feedback.get_recent_feedback(5); print(f'Average score: {feedback.get_average_score():.1f}/5.0\n') if feedback.get_average_score() else print('No feedback data available yet.\n'); [print(f'ID: {f[\"id\"]}\nScore: {f[\"score\"]}/5\nQuery: {f[\"user_input\"]}\nResponse: {f[\"model_response\"][:100]}...\nFeedback: {f[\"feedback_text\"]}\n---\n') for f in recent]"
            return_to_menu
            ;;
        5)
            echo
            echo "Thank you for using LangManus!"
            echo
            exit 0
            ;;
        *)
            echo "Invalid option, please try again."
            sleep 1
            ;;
    esac
done 