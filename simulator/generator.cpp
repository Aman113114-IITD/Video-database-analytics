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


bool compare(const entry &e1,const entry& e2){
	return e1.frame_id<e2.frame_id;
}


vector<entry> generate_stream() {
	vector<entry> traffic;
	// car
	Entity obj1(2,1,"RED",60,120,{0,200,150,300},{1770,200,1920,300});
	vector<entry> events1 = obj1.generate_object_stream();
	traffic.insert(traffic.end(),events1.begin(),events1.end());
	// car2 overtaking car1
	Entity obj2(2,2,"BLUE",70,100,{0,400,150,500},{1770,400,1920,500});
	vector<entry> events2 = obj2.generate_object_stream();
	traffic.insert(traffic.end(),events2.begin(),events2.end());
	// bicycle
	Entity obj3(1,3,"BLACK",60,150,{0,100,40,120},{1880,100,1920,120});
	vector<entry> events3 = obj3.generate_object_stream();
	traffic.insert(traffic.end(),events3.begin(),events3.end());
	// car3 going reverse
	Entity obj4(2,4,"GREEN",70,100,{1770,600,1920,700},{0,600,150,700});
	vector<entry> events4 = obj4.generate_object_stream();
	traffic.insert(traffic.end(),events4.begin(),events4.end());
	//car going forward and backward
	Entity obj5(2,5,"RED",50,90,{0,550,40,570},{980,550,1020,570});
	Entity obj6(2,5,"RED",91,110,{980,550,1020,570},{0,550,40,570});
	vector<entry> events5 = obj5.generate_object_stream();
	vector<entry> events6 = obj6.generate_object_stream();
	traffic.insert(traffic.end(),events5.begin(),events5.end());
	traffic.insert(traffic.end(),events6.begin(),events6.end());
	// cellphone 67 person 0
	Entity obj7(0,6,"BLACK",160,200,{0,10,20,30},{150,10,170,30});
	Entity obj8(0,7,"BLACK",150,250,{0,70,20,90},{180,70,200,90});
	Entity obj9(0,8,"BLACK",150,250,{0,110,20,130},{180,110,200,130});
	// with a cellphone
	Entity obj10(67,9,"BLUE",150,250,{5,115,15,125},{185,115,195,125});
	vector<entry> events7 = obj7.generate_object_stream();
	vector<entry> events8 = obj8.generate_object_stream();
	vector<entry> events9 = obj9.generate_object_stream();
	vector<entry> events10 = obj10.generate_object_stream();
	traffic.insert(traffic.end(),events7.begin(),events7.end());
	traffic.insert(traffic.end(),events8.begin(),events8.end());
	traffic.insert(traffic.end(),events9.begin(),events9.end());
	traffic.insert(traffic.end(),events10.begin(),events10.end());
	sort(traffic.begin(),traffic.end(),compare);
    return traffic;
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
