@startuml

start

partition Initialize {
    :load models and faces from database;
    :open file with video-capture;
}
while (video-capture open?) is (opened)
    :read current frame;
    repeat
        :detect faces;
        if (found faces?) then (yes)
        repeat
            :track face by distance
            and
            similarity length;
            if (tracked face?) then (yes)
                :mark as
                tracked;
            else (no)
                :mark as
                new track,
                tracked frames
                number: 0;
            endif
            :increase tracked frames
            number by 1;
            if (is center?) then (yes)
                :mark as
                center face;
            else (no)
            endif
            (A)
            detach
            (B)
            :paint face
            with colored rectangle;
        repeat while (more faces?)

        else (no)
        -[#blue]->
        endif
    repeat while (more frames?) -[#green]->
endwhile (closed)
end
detach
(A)
partition CuriousFaceID {
    if (tracked frames > 50) then (yes)
        :open popup with text-field
        and generate new faceID
        with inserted name;
    else (no)
        if (tracked frames > 10) then (yes)
        :generate new
        temporary faceID
        and mark as unknown;
        else (no)
        endif
    endif
}
partition Interactive_Learning {
    if (is learn-button activated?) then (yes)
        :rename tracked face
        with name in edit-field;
    else (no)
    endif
}

partition Identifying {
    :identify face
    by voting over faceIDs;
    if(faceID voting > 0.5) then (yes (identified))
        :mark as identified
        with labeled faceID;
        if(is center?) then (yes)
            :color is
            dark orange;
        else (no)
        :color is
        green;
        endif
        else (no)
        :mark as unknown;
        if (faceID voting == 0.0) then (yes (unknown))
            :color is white;
        else (no)
            if(is center?) then (yes)
            :color is
            bright orange;
            else (no)
            :color is
            yellow;
            endif
        endif
    endif
}
(B)
detach
@enduml