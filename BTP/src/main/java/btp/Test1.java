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
import org.apache.commons.lang3.tuple.Pair;

public class Test1 {

	private static Integer last_frameid = 0;

	public static void main(String[] args) throws Exception {
   	 
    	StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
    	Properties kafkaProps = new Properties();
    	kafkaProps.setProperty(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
    	kafkaProps.setProperty(ConsumerConfig.GROUP_ID_CONFIG, "flink-group");
    	KafkaSource<String> kafkaSource = KafkaSource.<String>builder()
        	.setBootstrapServers("localhost:9092")
        	.setTopics("events")
        	.setStartingOffsets(OffsetsInitializer.latest())
        	.setValueOnlyDeserializer(new SimpleStringSchema())
        	.build();
    	DataStream<String> kafkaStream = env.fromSource(kafkaSource,WatermarkStrategy.noWatermarks(), "Kafka Source");


		DataStream<Pair<Integer, Integer>> eventStream_single = kafkaStream.map(new MapFunction<String, Pair<Integer, Integer>>() {
        	@Override
        	public Pair<Integer, Integer> map(String value) {
            	value = value.substring(1, value.length() - 1);
            	String[] values = value.split(", ");
            	int frame_id = Integer.parseInt(values[0]);
            	int obj_id = Integer.parseInt(values[1]);
            	return Pair.of(frame_id,obj_id);
        	}
    	});

		// Pattern<Pair<Integer, Integer>, ?> pattern1 = Pattern.<Pair<Integer, Integer>>begin("q1",AfterMatchSkipStrategy.skipPastLastEvent())
        //         .where(SimpleCondition.of(value -> (value.getLeft()==1)))
        //         .oneOrMore().greedy();
		Pattern<Pair<Integer, Integer>, ?> pattern1 = Pattern.<Pair<Integer, Integer>>begin("q11")
                .where(SimpleCondition.of(value -> true))
                .followedBy("q12")
				.where(new IterativeCondition<Pair<Integer, Integer>>() {
					@Override
					public boolean filter(Pair<Integer, Integer> event, Context<Pair<Integer, Integer>> context) throws Exception {
						// System.out.println("START");
						// for ( Pair<Integer, Integer> val : context.getEventsForPattern("q11")) {
						// 	System.out.println(val.getLeft());
						// }
						// System.out.println("END");
						Pair<Integer, Integer> a = context.getEventsForPattern("q11").iterator().next();
						return a.getLeft()+1<event.getLeft() ;
					}
				})
				.followedBy("q13")
				.where(new IterativeCondition<Pair<Integer, Integer>>() {
					@Override
					public boolean filter(Pair<Integer, Integer> event, Context<Pair<Integer, Integer>> context) throws Exception {
						// System.out.println("START2");
						// for ( Pair<Integer, Integer> val : context.getEventsForPattern("q12")) {
						// 	System.out.println(val.getLeft());
						// }
						// System.out.println("END2");
						Pair<Integer, Integer> b = context.getEventsForPattern("q12").iterator().next();
						return (b.getLeft()-1>event.getLeft()) ;
					}
            	});

        Pattern<Pair<Integer, Integer>, ?> pattern2 = Pattern.<Pair<Integer, Integer>>begin("q2")
                .where(SimpleCondition.of(value -> (value.getLeft()==1)))
                .oneOrMore().greedy();
                // .times(2).greedy();
                // .until(SimpleCondition.of(value -> (value==2)));


		DataStream<Map<String,List<Pair<Integer, Integer>>>> resultStream1 = CEP.pattern(eventStream_single, pattern1).inProcessingTime()
        .select(new PatternSelectFunction<Pair<Integer, Integer>, Map<String,List<Pair<Integer, Integer>>>>() {
            @Override
            public Map<String,List<Pair<Integer, Integer>>> select(Map<String, List<Pair<Integer, Integer>>> pattern) throws Exception {
                Map<String, List<Pair<Integer, Integer>>> resultList = pattern;
                return resultList;
            }
        });

        // DataStream<Map<String,List<Pair<Integer, Integer>>>> resultStream2 = CEP.pattern(eventStream_single, pattern2).inProcessingTime()
        // .select(new PatternSelectFunction<Pair<Integer, Integer>, Map<String,List<Pair<Integer, Integer>>>>() {
        //     @Override
        //     public Map<String,List<Pair<Integer, Integer>>> select(Map<String, List<Pair<Integer, Integer>>> pattern) throws Exception {
        //         Map<String, List<Pair<Integer, Integer>>> resultList = pattern;
        //         return resultList;
        //     }
        // });

    	resultStream1.print();
        // resultStream2.print();
    	env.execute();
	}
}
