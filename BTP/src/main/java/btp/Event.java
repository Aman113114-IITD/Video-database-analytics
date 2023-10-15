/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package btp;

import java.util.Objects;

public class Event<A, B, C, D, E, F, G, H> {
    private final A frame_id;
    private final B obj_id;
    private final C obj_class;
    private final D color;
    private final E xmin;
    private final F ymin;
    private final G xmax;
    private final H ymax;

    public Event(A frame_id, B obj_id, C obj_class, D color, E xmin, F ymin, G xmax, H ymax) {
        this.frame_id = frame_id;
        this.obj_id = obj_id;
        this.obj_class = obj_class;
        this.color = color;
        this.xmin = xmin;
        this.ymin = ymin;
        this.xmax = xmax;
        this.ymax = ymax;
    }

    public A getframe_id() {
        return frame_id;
    }

    public B getobj_id() {
        return obj_id;
    }

    public C getobj_class() {
        return obj_class;
    }

    public D getcolor() {
        return color;
    }

    public E getxmin() {
        return xmin;
    }

    public F getymin() {
        return ymin;
    }

    public G getxmax() {
        return xmax;
    }

    public H getymax() {
        return ymax;
    }

    @Override
    public String toString() {
        return "(" + frame_id + ", " + obj_id + ", " + obj_class + ", " +
               color + ", " + xmin + ", " + ymin + ", " + xmax + ", " + ymax + ")";
    }
}

