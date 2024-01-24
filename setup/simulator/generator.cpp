#include <bits/stdc++.h>
using namespace std;

class entry {
    public:
        int frame_id;
        int object_id;
        int object_class;
        string color;
        float xmin;
        float ymin;
        float xmax;
        float ymax;

        string convert_to_string() {
            string ans="(";
            ans+=to_string(frame_id);
            ans+=", ";
            ans+=to_string(object_id);
            ans+=", ";
            ans+=to_string(object_class);
            ans+=", ";
            ans+=color;
            ans+=", ";
            ans+=to_string(xmin);
            ans+=", ";
            ans+=to_string(ymin);
            ans+=", ";
            ans+=to_string(xmax);
            ans+=", ";
            ans+=to_string(ymax);
            ans+=")";
            return ans;
        }
};

class Entity {

    private:
        float slope[4];

    public:
        int object_class;
        int object_id;
        string color;
        float initial_position[4];
        float final_position[4];
        int start_frame;
        int end_frame;

        Entity(int a,int b,string c,int d,int e,vector<float> ip,vector<float> fp) {
            object_class=a;
            object_id=b;
            color=c;
            start_frame=d;
            end_frame=e;
            for ( int i = 0 ; i < 4 ; i++ ) {
                initial_position[i]=ip[i];
                final_position[i]=fp[i];
            }
        }

        vector<entry> generate_object_stream() {
            for ( int i = 0 ; i < 4 ; i++ ) {
                slope[i]=final_position[i]-initial_position[i];
                slope[i]/=(end_frame-start_frame);
            }
            vector<entry> ans;
            for ( int i = start_frame ; i<= end_frame ; i++ ) {
                entry var;
                var.frame_id=i;
                var.object_id=object_id;
                var.object_class=object_class;
                var.color=color;
                var.xmin=initial_position[0]+slope[0]*(i-start_frame);
                var.ymin=initial_position[1]+slope[1]*(i-start_frame);
                var.xmax=initial_position[2]+slope[2]*(i-start_frame);
                var.ymax=initial_position[3]+slope[3]*(i-start_frame);
                ans.push_back(var);
            }
            return ans;
        }

};


vector<entry> generate_stream() {
    Entity obj1(2,1,"RED",60,120,{0,200,150,300},{1770,200,1920,300});
    vector<entry> ans;
    ans=obj1.generate_object_stream();
    Entity obj2(2,1,"RED",121,180,{1770,200,1920,300},{300,200,450,300});
    vector<entry> ans2=obj2.generate_object_stream();
    ans.insert(ans.end(),ans2.begin(),ans2.end());
    return ans;
}

int main() {
    vector<entry> video1=generate_stream();
    ofstream out("logs1.txt");
    for ( int i = 0 ; i < video1.size() ; i++ ) {
        string var = video1[i].convert_to_string();
        out<<var<<"\n";
    }
    out.close();
}