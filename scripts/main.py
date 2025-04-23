import os

def run_script(script_name):
    script_path = os.path.join("", script_name)
    os.system(f"python {script_path}")

def main():
    while True:
        print("\n=== Oculus_HGR Menu ===")
        print("1. Add new gesture samples")
        print("2. Merge all gesture data")
        print("3. Train gesture recognition model")
        print("4. Check known gesture labels")
        print("5. Run real-time prediction")
        print("6. Exit")

        choice = input("Select an option (1-6): ").strip()

        if choice == "1":
            run_script("add_samples.py")
        elif choice == "2":
            run_script("merge_data.py")
        elif choice == "3":
            run_script("train_model.py")
        elif choice == "4":
            run_script("check_labels.py")
        elif choice == "5":
            run_script("run_realtime.py")
        elif choice == "6":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
