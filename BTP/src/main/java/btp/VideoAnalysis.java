package btp;
import org.apache.flink.api.common.functions.MapFunction;
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
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.api.java.functions.KeySelector;

public class VideoAnalysis {

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

    	DataStream<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>> eventStream = kafkaStream.map(new MapFunction<String, Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>() {
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

   	 Pattern<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>, ?> pattern = Pattern.<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>begin("redObject")
   	 .where(
        new SimpleCondition<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>() {
            @Override
            public boolean filter(Event<Integer, Integer, Integer, String, Float, Float, Float, Float> event) throws Exception {
                return event.getcolor().equals("'RED'");
            }
        }
	 )
   	 .next("blackObject")
   	 .where(
   		 new IterativeCondition<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>() {
   			 @Override
   			 public boolean filter(
   				 Event<Integer, Integer, Integer, String, Float, Float, Float, Float> blackEvent,
   				 Context<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>> context
   			 ) throws Exception {
   				 // Check if the ymin of blackObject is less than ymin of redObject
   				 Event<Integer, Integer, Integer, String, Float, Float, Float, Float> redEvent = context.getEventsForPattern("redObject").iterator().next();
   				 return blackEvent.getymin() > redEvent.getymin() && blackEvent.getcolor().equals("'BLACK'");
   			 }
   		 }
   	 );

    	// Apply the pattern to the event stream
   	 PatternStream<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>> patternStream = CEP.pattern(eventStream, pattern).inProcessingTime();

   	 // Select and print the matched patterns
   	 DataStream<Tuple2<List<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>, List<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>>> resultStream = patternStream.select(
   		 new PatternSelectFunction<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>, Tuple2<List<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>, List<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>>>() {
   			 @Override
   			 public Tuple2<List<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>, List<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>> select(Map<String, List<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>>> pattern) throws Exception {
   				 List<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>> redObject = pattern.get("redObject");
   				 List<Event<Integer, Integer, Integer, String, Float, Float, Float, Float>> blackObject = pattern.get("blackObject");
   				 return new Tuple2<>(redObject, blackObject);
   			 }
   		 }
   	 );
    	resultStream.print();
    	env.execute();
	}
}




