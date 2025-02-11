#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

using namespace std; 

class Animals {
    public: 
        string species;
        string name;
        string brain;
        int id;
        int status;
        int mating;
        int speed; 
        int eyesight;
        int fov;
};

Animals animal_placeholder;

Animals matrix[3][4]  = {
    {animal_placeholder, animal_placeholder, animal_placeholder, animal_placeholder}, 
    {animal_placeholder, animal_placeholder, animal_placeholder, animal_placeholder}, 
    {animal_placeholder, animal_placeholder, animal_placeholder, animal_placeholder}, 
};

vector <Animals> list;

int main() {
    for (int i = 0; i <= 11; i++) {
        Animals temp;
        temp.name = "sheep" + to_string(i);
        temp.id = i;
        list.push_back(temp);
        int row = i / 4;
        int column = i % 4;
        matrix[row][column] = temp;
    }

    for (int row = 0; row <= 2; row ++ ) {
        for (int column = 0; column <= 3; column++) {
            Animals temp = matrix[row][column];
            cout << temp.name;
        }
        cout << endl;
    }
    
    //cout << sheep.name;
    return 0;;
}


