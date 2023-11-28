package btp;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.cep.CEP;
import org.apache.flink.cep.pattern.Pattern;
import org.apache.flink.cep.PatternSelectFunction;
import org.apache.flink.cep.pattern.conditions.SimpleCondition;
import org.apache.flink.cep.pattern.conditions.IterativeCondition;
import org.apache.flink.cep.PatternStream;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.common.serialization.StringDeserializer;
import java.util.Properties;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.regex.Matcher;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.util.Collector;
import org.apache.flink.cep.nfa.aftermatch.AfterMatchSkipStrategy;

public class Backward {

	private static Integer last_frameid = 0;

	public static void main(String[] args) throws Exception {
   	 
    	StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
    	Properties kafkaProps = new Properties();
    	kafkaProps.setProperty(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
    	kafkaProps.setProperty(ConsumerConfig.GROUP_ID_CONFIG, "flink-group");
    	// FlinkKafkaConsumer<String> kafkaConsumer = new FlinkKafkaConsumer<>("events", new SimpleStringSchema(), properties);
    	KafkaSource<String> kafkaSource = KafkaSource.<String>builder()
        	.setBootstrapServers("localhost:9092")
        	.setTopics("events")
        	.setStartingOffsets(OffsetsInitializer.latest())
        	.setValueOnlyDeserializer(new SimpleStringSchema())
        	.build();
    	DataStream<String> kafkaStream = env.fromSource(kafkaSource,WatermarkStrategy.noWatermarks(), "Kafka Source");

		ArrayList<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>> frame_history = new ArrayList<>();

    	// DataStream<PairBB> eventStream = kafkaStream.flatMap(new FlatMapFunction<String, PairBB>() {
		// 	@Override
		// 	public void flatMap(String value, Collector<PairBB> out) {
        //     	value = value.substring(1, value.length() - 1);
        //     	String[] values = value.split(", ");
        //     	int frame_id = Integer.parseInt(values[0]);
		// 		// System.out.println(frame_id);
        //     	int obj_id = Integer.parseInt(values[1]);
        //     	int obj_class = Integer.parseInt(values[2]);
        //     	String color = values[3];
        //     	float xmin = Float.parseFloat(values[4]);
        //     	float ymin = Float.parseFloat(values[5]);
        //     	float xmax = Float.parseFloat(values[6]);
        //     	float ymax = Float.parseFloat(values[7]);
        //     	Event<Integer, Integer, Integer, String, Float, Float, Float, Float> cur_obj = new Event<>(frame_id, obj_id, obj_class, color, xmin, ymin, xmax, ymax);
		// 		if (frame_history.isEmpty()) {
		// 			frame_history.add(cur_obj);
		// 			last_frameid=cur_obj.getframe_id();
		// 		}
		// 		else {
		// 			if (last_frameid==cur_obj.getframe_id()) {
		// 				for ( int i = 0 ; i < frame_history.size() ; i++) {
		// 					PairBB var = new PairBB(frame_history.get(i),cur_obj);
		// 					out.collect(var);
		// 				}
		// 				frame_history.add(cur_obj);
		// 			}
		// 			else {
		// 				frame_history.clear();
		// 				last_frameid=cur_obj.getframe_id();
		// 				frame_history.add(cur_obj);
		// 			}
		// 		}
        // 	}
    	// });


		DataStream<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>> eventStream_single = kafkaStream.map(new MapFunction<String, Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>() {
        	@Override
        	public Event<Integer, Integer, Integer, String, Float, Float, Float, Float> map(String value) {
            	value = value.substring(1, value.length() - 1);
            	String[] values = value.split(", ");
            	int frame_id = Integer.parseInt(values[0]);
            	int obj_id = Integer.parseInt(values[1]);
            	int obj_class = Integer.parseInt(values[2]);
            	String color = values[3];
            	float xmin = Float.parseFloat(values[4]);
            	float ymin = Float.parseFloat(values[5]);
            	float xmax = Float.parseFloat(values[6]);
            	float ymax = Float.parseFloat(values[7]);
            	return new Event<>(frame_id, obj_id, obj_class, color, xmin, ymin, xmax, ymax);
        	}
    	});

		// Pattern<PairBB, ?> pattern = Pattern.<PairBB>begin("start")
        //         .where(SimpleCondition.of(value -> (value.getobject1().getobj_class()==5 && value.getobject2().getobj_class()==5)));

		// Pattern<PairBB, ?> pattern = Pattern.<PairBB>begin("start")
        //         .where(SimpleCondition.of(value -> (value.getobject1().getobj_class()==5 && value.getobject2().getobj_class()==5)));

		Pattern<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>, ?> pattern_single = Pattern.<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>begin("a")
            .where(new SimpleCondition<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>() {
                @Override
                public boolean filter(Event<Integer, Integer, Integer, String, Float, Float, Float, Float> event) {
                    return true;
					// return true;
                }
            })
            .followedBy("b")
            .where(new IterativeCondition<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>() {
                @Override
                public boolean filter(Event<Integer, Integer, Integer, String, Float, Float, Float, Float> event, Context<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>> context) throws Exception {
                    Event<Integer, Integer, Integer, String, Float, Float, Float, Float> a = context.getEventsForPattern("a").iterator().next();
					float cenxo = (a.getxmax()+a.getxmin())/2;
					float cenyo = (a.getymax()+a.getymin())/2;
					float cenxn = (event.getxmax()+event.getxmin())/2;
					float cenyn = (event.getymax()+event.getymin())/2;
					float width = (a.getxmax()-a.getxmin());
					float height = (a.getymax()-a.getymin());
					if ((event.getframe_id()>a.getframe_id()) && (event.getobj_id()==a.getobj_id()) && ((cenxo-cenxn)>=width*0.2) && ((cenyo-cenyn)>=height*0.2)) {
						System.out.println("A car is moving in reverse direction");
					}
                    return  (event.getframe_id()>a.getframe_id()) && (event.getobj_id()==a.getobj_id()) && ((cenxo-cenxn)>=width*0.2) && ((cenyo-cenyn)>=height*0.2) ;
                }
            });

		// Pattern<PairBB, ?> pattern = Pattern.<PairBB>begin("a")
        //     .where(new SimpleCondition<PairBB>() {
        //         @Override
        //         public boolean filter(PairBB event) {
		// 			// System.out.println("this");
		// 			// return true;
		// 			Event<Integer, Integer, Integer, String, Float, Float, Float, Float> cp=new Event<>(1,1,1,"a",0.0f,0.0f,0.0f,0.0f);
		// 			Event<Integer, Integer, Integer, String, Float, Float, Float, Float> per=new Event<>(1,1,1,"a",0.0f,0.0f,0.0f,0.0f);
		// 			if ((event.getobject1().getobj_class()==1) && (event.getobject2().getobj_class()==0)) {
		// 				// return true;
		// 				cp=event.getobject1();
		// 				per=event.getobject2();
		// 			}
		// 			else if ((event.getobject1().getobj_class()==0) && (event.getobject2().getobj_class()==1)) {
		// 				// return true;
		// 				cp=event.getobject2();
		// 				per=event.getobject1();
		// 			}
		// 			else {
		// 				return false;
		// 			}
		// 			// return (cp.getxmax()<per.getxmax()) && (cp.getxmin()>per.getxmin()) && (cp.getymax()<per.getymax()) && (cp.getymin()>per.getymin());
		// 			float cenx=(cp.getxmin()+cp.getxmax())/2;
		// 			float ceny=(cp.getymin()+cp.getymax())/2;
		// 			if ((cp.getxmin()<=per.getxmin()) && (cp.getxmax()>=per.getxmax())) {
		// 				System.out.println(per.getframe_id() + " Riding : "+per.getobj_id()+" "+cp.getobj_id());
		// 			}
		// 			else if ( Math.abs(cenx-per.getxmax())<=Math.abs(per.getxmax()-per.getxmin())){
		// 				System.out.println(per.getframe_id() + " Not Riding : "+per.getobj_id()+" "+cp.getobj_id());
		// 			}
		// 			// return (cenx<per.getxmax()) && (cenx>per.getxmin()) && (ceny<per.getymax()) && (ceny>per.getymin());
		// 			return false;
        //         }
        //     });

		// Pattern<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>, ?> pattern_single = Pattern.<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>begin("e1",AfterMatchSkipStrategy.skipPastLastEvent())
		// 	.where(new SimpleCondition<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>() {
		// 		@Override
		// 		public boolean filter(Event<Integer, Integer, Integer, String, Float, Float, Float, Float> event) {
		// 			return event.getobj_class()==2;
		// 		}
		// 	})
		// 	.times(150);
		
        // DataStream<Map<String,List<PairBB>>> resultStream = CEP.pattern(eventStream, pattern).inProcessingTime()
        // .select(new PatternSelectFunction<PairBB, Map<String,List<PairBB>>>() {
        //     @Override
        //     public Map<String,List<PairBB>> select(Map<String, List<PairBB>> pattern) throws Exception {
        //         Map<String, List<PairBB>> resultList = pattern;
        //         return resultList;
        //     }
        // });

		DataStream<Map<String,List<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>>> resultStream = CEP.pattern(eventStream_single, pattern_single).inProcessingTime()
        .select(new PatternSelectFunction<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>, Map<String,List<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>>>() {
            @Override
            public Map<String,List<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>> select(Map<String, List<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>> pattern) throws Exception {
                Map<String, List<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>> resultList = pattern;
                return resultList;
            }
        });

    	resultStream.print();
    	env.execute();
	}
}
