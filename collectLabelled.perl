'''
    
    Copyright (C) 2017 Luca Surace - University of Calabria, Plymouth University
    
    This file is part of Deemotions. Deemotions is an Emotion Recognition System
    based on Deep Learning method.

    Deemotions is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Deemotions is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Deemotions.  If not, see <http://www.gnu.org/licenses/>.
    
    -----------------------------------------------------------------------
    
    This code selects the peak frames of each valid sequence of CK+ database.
    A sequence is valid if there is its associated emotion file in the corresponding folder.
    It also selects the related landmark files.
    
'''

mkdir "ExtractedPeakFramesLandmarks"; 
opendir (DIR, "./cohn-kanade-images/");
opendir (DIR, "./Emotion/");
while ($subject = readdir(DIR)) {
	next if ($subject =~ /^..?$/);
	chomp $subject;
    push (@subjects, $subject);
}
@subjects = sort {$a cmp $b} @subjects;

foreach $subject (@subjects) {
    opendir (SUBDIR, "./cohn-kanade-images/$subject");
	while ($sequence = readdir(SUBDIR)) {
		next if ($sequence =~ /^..?$/);
		next if ($sequence =~ /.DS_Store/);
		chomp $sequence;
    	push (@sequences, $sequence);
	}
	
	
	
	@sequences = sort {$a cmp $b} @sequences;
	
	foreach $sequence (@sequences) {
		
		opendir (SUBSUBDIR, "./cohn-kanade-images/$subject/$sequence");
		while ($frame = readdir(SUBSUBDIR)) {
			next if ($frame =~ /^..?$/);
			chomp $frame;
    		push (@frames, $frame);
    		@frames = sort {$a cmp $b} @frames;
		}
		if (!is_folder_empty("./Emotion/$subject/$sequence")) {
			qx(cp "./cohn-kanade-images/$subject/$sequence/$frames[$#frames]" ./ExtractedPeakFramesLandmarks);
			$smallFrame = substr($frames[$#frames],0,length($frames[$#frames])-4);
			qx(cp "./Landmarks/$subject/$sequence/$smallFrame"_landmarks.txt ./ExtractedPeakFramesLandmarks);
		}
		
		@frames = ();
	}
	
	@sequences = ();
}

closedir(DIR);

sub is_folder_empty {
    my $dirname = shift;
    opendir(my $dh, $dirname) or mkdir "$dirname";
    return scalar(grep { $_ ne "." && $_ ne ".." } readdir($dh)) == 0;
}
